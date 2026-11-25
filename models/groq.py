from __future__ import annotations

import json
from typing import Any

import httpx

from .base import ChatResponse, Message, ProviderError, ToolCall


class GroqProvider:
    """Groq OpenAI-compatible chat-completions adapter."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        host: str = "https://api.groq.com/openai/v1",
        timeout: float = 180.0,
    ):
        self.model = model
        self.api_key = api_key
        self.host = host.rstrip("/")
        self.timeout = timeout

    async def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ChatResponse:
        if not self.api_key:
            raise ProviderError("Groq requires GROQ_API_KEY. Create an API key and set it in your environment.")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._message(message) for message in messages],
        }
        if tools:
            payload["tools"] = tools

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.host}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
            choice = data["choices"][0]["message"]
            return ChatResponse(
                message=Message(
                    role=choice.get("role", "assistant"),
                    content=choice.get("content") or "",
                    tool_calls=self._tool_calls(choice.get("tool_calls") or []),
                ),
                done=True,
            )
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Groq request failed: {exc}") from exc

    @staticmethod
    def _message(message: Message) -> dict[str, Any]:
        result: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id
            if message.tool_name:
                result["name"] = message.tool_name
        return result

    @staticmethod
    def _tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
        calls = []
        for index, call in enumerate(raw_calls):
            function = call.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError as exc:
                    raise ProviderError(f"Groq returned invalid arguments for {function.get('name', '')}") from exc
            if not isinstance(arguments, dict):
                raise ProviderError(f"Groq returned non-object arguments for {function.get('name', '')}")
            calls.append(ToolCall(
                id=call.get("id") or f"groq-{index}",
                name=function.get("name", ""),
                arguments=arguments,
            ))
        return calls
