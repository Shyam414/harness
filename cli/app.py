from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from context.manager import ContextManager
from core.agent import Agent
from core.registry import ToolRegistry
from models.base import Message, ProviderError
from models.ollama import OllamaProvider
from runtime.manager import RuntimeManager
from tools.filesystem import register_filesystem
from tools.git import register_git
from tools.terminal import register_terminal


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without overriding existing environment values."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _clip(text: str, limit: int = 200) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _summarize(arguments: dict) -> str:
    """One-line description of a tool call; never prints file contents."""
    return _clip(arguments.get("path") or arguments.get("command") or str(arguments.get("pid", "")))


def _summarize_result(raw: str) -> str:
    try:
        parsed = json.loads(raw)
    except ValueError:
        return "[result] (unreadable)"
    if not parsed.get("ok"):
        return f"[result] FAILED: {_clip(parsed.get('error', ''), 300)}"
    result = parsed.get("result") or {}
    if isinstance(result.get("returncode"), int) or result.get("timed_out"):
        output = "\n".join(part for part in (result.get("stdout", ""), result.get("stderr", "")) if part)
        status = "timed out" if result.get("timed_out") else f"exit {result['returncode']}"
        if result.get("shell"):
            status += f" in {result['shell']}"
        if result.get("terminal_closed"):
            status += "; visible terminal closed"
        return f"[result] {status}" + (f":\n{_clip(output, 4000)}" if output else "")
    if result.get("started") is True:
        shell = result.get("shell")
        suffix = f" in {shell}" if shell else ""
        return f"[result] started PID {result.get('pid')}{suffix}"
    if result.get("stopped") is True:
        return f"[result] stopped PID {result.get('pid')}"
    if isinstance(result.get("deleted_count"), int):
        return f"[result] deleted {result['deleted_count']} file(s)"
    if result.get("created") is True:
        return "[result] created"
    if result.get("created") is False:
        return "[result] updated" if "bytes" in result else "[result] already exists"
    return "[result] ok"


def _event(kind: str, data: dict) -> None:
    if kind == "token":
        print(data["text"], end="", flush=True)
    elif kind == "tool_start":
        print(f"\n[tool] {data['name']} {_summarize(data.get('arguments') or {})}", flush=True)
    elif kind == "tool_result":
        print(_summarize_result(data["result"]), flush=True)
    elif kind == "error":
        print(f"[error] {data['message']}", flush=True)


def _load_system_prompt(workspace: Path, shell_info: dict[str, str]) -> str:
    template = Path(__file__).with_name("system_prompt.txt").read_text(encoding="utf-8")
    tool_prompt = Path(__file__).with_name("tools_prompt.txt").read_text(encoding="utf-8")
    shell_instruction = (
        f"Detected shell for terminal_run, terminal_start, and terminal_open: "
        f"{shell_info['name']} ({shell_info['executable']}). Write commands using this shell's syntax. "
        f"Curl shell policy: {shell_info.get('curl_shell', shell_info['name'])}."
    )
    return f"{template.format(workspace=workspace)}\n\n{shell_instruction}\n\n{tool_prompt}"


async def _chat(args) -> None:
    workspace = Path(args.workspace).resolve()
    sessions = Path(args.sessions)
    if not sessions.is_absolute():
        sessions = workspace / sessions

    runtime = RuntimeManager(workspace)
    registry = ToolRegistry()
    register_filesystem(registry, runtime)
    register_git(registry, runtime)
    register_terminal(registry, runtime)

    context = ContextManager(sessions, args.session)
    system_message = Message(role="system", content=_load_system_prompt(workspace, runtime.shell_info()))
    existing = next((i for i, m in enumerate(context.messages) if m.role == "system"), None)
    if existing is None:
        context.messages.insert(0, system_message)
    else:
        context.messages[existing] = system_message

    provider = OllamaProvider(args.model, args.host)
    await provider.ensure_model()
    capabilities = (await provider.model_info()).get("capabilities", [])
    if "tools" not in capabilities:
        logging.getLogger("agent").warning("model=%s does not advertise native tool calling; file creation may not work", args.model)

    agent = Agent(provider, registry, context, max_turns=args.max_turns, on_event=_event)
    curl_shell = runtime.shell_info().get("curl_shell")
    curl_label = f" | curl: {curl_shell}" if curl_shell else ""
    print(f"Local coding agent | model: {args.model} | shell: {runtime.shell_name}{curl_label} | workspace: {workspace}")

    async def run_prompt(prompt: str) -> None:
        result = await agent.run(prompt)
        print(f"\n\n[completed in {result.turns} turns; {result.tool_calls} tool calls]")
        if result.response and result.tool_calls == 0:
            print(result.response)

    try:
        if args.prompt:
            await run_prompt(" ".join(args.prompt))
            return
        print("Describe what to create or change, or type /exit to quit.\n")
        # Pasting inserts the complete clipboard payload into the prompt buffer.
        # It does not submit: Enter submits the full draft, including pasted
        # lines and any additional text typed afterward.
        bindings = KeyBindings()

        @bindings.add(Keys.BracketedPaste, eager=True)
        def _insert_paste(event) -> None:
            paste_data = event.key_sequence[0].data or ""
            event.current_buffer.insert_text(paste_data)

        prompt_session = PromptSession(multiline=False, key_bindings=bindings)
        while True:
            try:
                prompt = (await prompt_session.prompt_async("You> ")).strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nInput cancelled.")
                continue
            if prompt.lower() in ("/exit", "/quit", "exit", "quit"):
                break
            if prompt:
                await run_prompt(prompt)
    finally:
        context.save()
        await runtime.close()


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Local AI agent harness (file and folder creation/update)")
    parser.add_argument("prompt", nargs="*", help="optional one-shot task; omit for an interactive session")
    parser.add_argument("--workspace", default=os.getenv("HARNESS_WORKSPACE", "."))
    parser.add_argument("--sessions", default=os.getenv("HARNESS_SESSION_DIR", ".harness/sessions"))
    parser.add_argument("--session", default="default")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--max-turns", type=int, default=int(os.getenv("HARNESS_MAX_TURNS", "40")), help="maximum model/tool rounds per task")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        asyncio.run(_chat(args))
    except KeyboardInterrupt:
        pass
    except ProviderError as exc:
        raise SystemExit(f"[error] {exc}")


if __name__ == "__main__":
    main()
