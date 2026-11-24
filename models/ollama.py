
from __future__ import annotations

from typing import Any
import asyncio
import httpx
import json
import logging
import os
import shutil
import subprocess

from .base import ChatResponse, Message, ProviderError, ToolCall

log = logging.getLogger("ollama")


class OllamaProvider:
    """Ollama /api/chat provider with native tool calling."""

    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", timeout: float = 180.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    async def ensure_server(self, on_status=print) -> None:
        """Make sure the Ollama server answers; start `ollama serve` in the background if it does not."""
        async def up() -> bool:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    return (await client.get(f"{self.host}/api/tags")).status_code == 200
            except httpx.HTTPError:
                return False

        if await up():
            return
        exe = shutil.which("ollama")
        if not exe:
            raise ProviderError("Ollama is not running and the 'ollama' command was not found. Install it from https://ollama.com/download")
        on_status("Ollama is not running - starting it...")
        kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000 | 0x00000008  # CREATE_NO_WINDOW | DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen([exe, "serve"], **kwargs)
        for _ in range(60):
            if await up():
                return
            await asyncio.sleep(0.5)
        raise ProviderError("Could not start Ollama. Start it manually with: ollama serve")

    async def ensure_model(self, on_status=print) -> None:
        """Check the local Ollama library and pull the selected model if needed."""
        await self.ensure_server(on_status)
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.get(f"{self.host}/api/tags")
                response.raise_for_status()

                models = response.json().get("models", [])
                installed = {item.get("name") for item in models}

                if self.model in installed or any(
                    name and name.split(":")[0] == self.model.split(":")[0]
                    and ":" not in self.model
                    for name in installed
                ):
                    return

                on_status(f"Model {self.model} is not installed. Pulling from Ollama...")

                last_percent = -1

                async with client.stream(
                    "POST",
                    f"{self.host}/api/pull",
                    json={"name": self.model, "model": self.model, "stream": True}
                ) as pull:
                    pull.raise_for_status()

                    async for line in pull.aiter_lines():
                        if not line:
                            continue

                        status = json.loads(line)

                        if status.get("error"):
                            raise ProviderError(
                                f"Ollama could not pull {self.model}: {status['error']}"
                            )

                        message = status.get("status", "")
                        total = status.get("total")
                        completed = status.get("completed")

                        if total and completed is not None:
                            percent = completed * 100 // total

                            if percent != last_percent:
                                on_status(f"Downloading {self.model}: {percent}%")
                                last_percent = percent

                        elif message in (
                            "pulling manifest",
                            "verifying sha256 digest",
                            "writing manifest",
                            "success"
                        ):
                            on_status(message)

                on_status(f"Model {self.model} is ready.")

        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise ProviderError(
                f"Could not check or pull model from Ollama at {self.host}: {exc}"
            ) from exc

    async def model_info(self) -> dict[str, Any]:
        """Return Ollama's model metadata without assuming tool support."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.host}/api/show",
                    json={"name": self.model, "model": self.model}
                )
                response.raise_for_status()
                data = response.json()

            capabilities = data.get("capabilities", [])

            return {
                "capabilities": capabilities if isinstance(capabilities, list) else [],
                "details": data.get("details", {})
            }

        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(
                f"Could not inspect Ollama model {self.model}: {type(exc).__name__}"
            ) from exc

    async def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._message(m) for m in messages],
            "stream": False
        }

        if tools:
            payload["tools"] = tools

        log.debug(
            "model=%s registered_tools=%d tools_included=%s",
            self.model, len(tools), bool(tools)
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.host}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()

            raw = data.get("message", {})
            raw_calls = raw.get("tool_calls") or []
            calls = self._tool_calls(raw_calls)

            log.debug(
                "assistant_tool_calls=%s requested_tool_calls=%d",
                bool(calls), len(calls)
            )

            return ChatResponse(
                message=Message(
                    role=raw.get("role", "assistant"),
                    content=raw.get("content", ""),
                    tool_calls=calls
                ),
                done=data.get("done", True)
            )

        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc

    async def chat_stream(self, messages: list[Message], tools: list[dict[str, Any]], on_token) -> ChatResponse:
        """Stream response text to the caller; tool calls arrive in the final chunk."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._message(m) for m in messages],
            "stream": True
        }

        if tools:
            payload["tools"] = tools

        log.debug(
            "model=%s registered_tools=%d tools_included=%s",
            self.model, len(tools), bool(tools)
        )

        parts: list[str] = []
        final: dict[str, Any] = {}
        streamed_calls: dict[int, dict[str, Any]] = {}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.host}/api/chat",
                    json=payload
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        chunk = json.loads(line)
                        if chunk.get("error"):
                            raise ProviderError(f"Ollama error: {chunk['error']}")
                        raw = chunk.get("message", {})

                        content = raw.get("content") or ""

                        if content:
                            parts.append(content)
                            on_token(content)

                        for index, call in enumerate(raw.get("tool_calls", [])):
                            fn = call.get("function", {})
                            streamed_calls[int(fn.get("index", index))] = call

                        if chunk.get("done"):
                            final = raw

            calls = self._tool_calls(
                list(streamed_calls.values()) or final.get("tool_calls") or []
            )

            log.debug(
                "assistant_tool_calls=%s requested_tool_calls=%d",
                bool(calls), len(calls)
            )

            return ChatResponse(
                message=Message(
                    role=final.get("role", "assistant"),
                    content="".join(parts),
                    tool_calls=calls
                ),
                done=True
            )

        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise ProviderError(
                f"Ollama streaming request failed: {exc}"
            ) from exc

    @staticmethod
    def _tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
        calls = []

        for index, call in enumerate(raw_calls):
            fn = call.get("function", {})
            args = fn.get("arguments", {})

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError as exc:
                    raise ProviderError(
                        f"Ollama returned invalid arguments for tool "
                        f"{fn.get('name', '')}: {exc}"
                    ) from exc

            if not isinstance(args, dict):
                raise ProviderError(
                    f"Ollama returned non-object arguments for tool {fn.get('name', '')}"
                )

            calls.append(
                ToolCall(
                    id=call.get("id") or f"ollama-{index}",
                    name=fn.get("name", ""),
                    arguments=args
                )
            )

        return calls

    @staticmethod
    def _message(message: Message) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": message.role,
            "content": message.content
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "function": {
                        "name": c.name,
                        "arguments": c.arguments
                    }
                }
                for c in message.tool_calls
            ]

        if message.tool_call_id:
            result["tool_name"] = message.tool_name or message.tool_call_id

        return result
