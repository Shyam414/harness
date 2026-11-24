from __future__ import annotations

import asyncio
import ast
import json
import logging
import random
import re
import shlex
from dataclasses import dataclass
from typing import Callable

from context.manager import ContextManager
from models.base import ChatProvider, Message, ToolCall
from .registry import ToolRegistry

log = logging.getLogger("harness.agent")

# Only action verbs indicate a workspace change request. Words like "file" or
# "folder" often appear in questions and must not enable tools by themselves.
_ACTION_INTENT = re.compile(
    r"^\s*(?:(?:please|now|hey|can\s+(?:you|u)|could you|would you|i want you to)\s+){0,3}"
    r"(?:make|build|crea(?:t?e|ted|ting)|generate|scaffold|write|add|edit|update|change|modify|fix|save|delete|remove|set\s+up|setup)\b",
    re.IGNORECASE,
)
# Users often state the diagnostic request first and the requested change later
# (for example, "see why X fails and update the code"). The anchored matcher
# above intentionally avoids treating questions as authorization, so recognize
# explicit change-verb + workspace-object phrases anywhere in the prompt too.
_EXPLICIT_CHANGE_INTENT = re.compile(
    r"\b(?:write|add|edit|update|change|modify|fix|save|delete|remove)\s+"
    r"(?:the\s+)?(?:codes?|files?|scripts?|app(?:lication)?|project|workspace|folder|directory)\b",
    re.IGNORECASE,
)
_INSPECT_INTENT = re.compile(
    r"\b(list|show|see|view|display|what(?:'s| is)?|which|find)\b.{0,120}\b(files?|folders?|directories|contents?)\b"
    r"|\b(files?|folders?|directories)\b.{0,120}\b(list|show|see|view|display|there|inside|in this)\b"
    r"|\b(path|location)\b.{0,100}\b(file|folder|directory|current|this)\b"
    r"|\b(file|folder|directory|current)\b.{0,100}\b(path|location)\b",
    re.IGNORECASE,
)
_RUN_INTENT = re.compile(
    r"^\s*(?:(?:please|now|can\s+(?:you|u)|could you|would you)\s+){0,3}"
    r"(?:run|execute|launch|start)\b|\b(run|execute)\s+(?:it|this|that|the\s+\S+)",
    re.IGNORECASE,
)
_VISIBLE_TERMINAL_INTENT = re.compile(
    r"\b(?:open|launch|use)\b.{0,60}\b(?:another|separate|new|visible)?\s*terminal\b"
    r"|\b(?:in|from)\s+(?:the\s+)?(?:other|another|separate|new)\s+terminal\b"
    r"|\bterminal\s+window\b",
    re.IGNORECASE,
)
_RUN_FILE_INTENT = re.compile(
    r"\b(?:run|execute|launch|start)\b.{0,80}\b[\w./\\-]+\.(?:py|js|ts|sh|bat|cmd|ps1|exe)\b",
    re.IGNORECASE,
)
_PATH_INTENT = re.compile(r"\b(path|location)\b", re.IGNORECASE)
_PROCESS_LIST_INTENT = re.compile(
    r"\b(?:list|show|see|check|what(?:'s|\s+is)?|which|find)\b.{0,120}"
    r"\b(?:running|active|started)\b.{0,50}\b(?:process(?:es)?|apps?|servers?|commands?)\b"
    r"|\bwhat(?:'s|\s+is)?\s+running\b.{0,80}\b(?:folder|directory|workspace|here)\b"
    r"|\b(?:process(?:es)?|apps?|servers?)\b.{0,80}\b(?:running|active)\b",
    re.IGNORECASE,
)
_PROCESS_STOP_INTENT = re.compile(
    r"^\s*(?:(?:please|now|can\s+(?:you|u)|could you|would you)\s+){0,3}"
    r"(?:stop|kill|terminate)\s+(?:(?:the|this|that|running)\s+){0,2}"
    r"(?:process|app|server|script|pid)\b",
    re.IGNORECASE,
)
_CURL_REQUEST_INTENT = re.compile(
    r"\b(?:use|run|execute|send|fetch|get)\b.{0,100}\b(?:curl|endpoint|api\s+request)\b",
    re.IGNORECASE,
)
_RUN_CONFIRMATION = re.compile(
    r"^\s*(?:yes|yeah|yep|sure|okay|ok|please|go ahead|proceed)(?:\s*[,!.]?\s*(?:run it|do it|proceed|go ahead|execute it))?\s*[!.]?\s*$",
    re.IGNORECASE,
)


def _redact_sensitive_text(text: str) -> str:
    """Mask credential values before persisting prompts or synthetic tool calls."""
    text = re.sub(
        r'(?i)((?:"|\b)(?:password|token|secret|api[_-]?key)(?:"|\b)\s*[:=]\s*)(?:"(?:\\.|[^"\\])*"|\S+)',
        r"\1[redacted]",
        text,
    )
    return re.sub(r"(?i)(--?(?:password|token|secret|api[-_]?key)\s+)\S+", r"\1[redacted]", text)


def _curl_login_command(prompt: str) -> str | None:
    """Build a Bash-safe curl login command from an explicit URL/email/password request."""
    url_match = re.search(r"https?://[^\s\"'<>]+", prompt, re.IGNORECASE)
    email_match = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", prompt, re.IGNORECASE)
    password_match = re.search(
        r"\bpassword\s*[:=]\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s,;]+))",
        prompt,
        re.IGNORECASE,
    )
    if not (url_match and email_match and password_match):
        return None
    url = url_match.group(0).rstrip(".,;:!?)]")
    password = next((value for value in password_match.groups() if value is not None), "")
    payload = json.dumps({"email": email_match.group(0), "password": password}, separators=(",", ":"))
    return (
        f"curl -i -X POST {shlex.quote(url)} "
        f"-H {shlex.quote('Content-Type: application/json')} -d {shlex.quote(payload)}"
    )


def _invocation_end(text: str, opening: int) -> int | None:
    """Find a function call's closing parenthesis while respecting quoted strings."""
    depth = 0
    quote = None
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _explicit_read_invocation(prompt: str) -> tuple[str, dict] | None:
    """Parse explicit read-only tool syntax such as git_status(path='.')."""
    names = ("git_status", "git_diff", "fs_list", "process_list")
    for name in names:
        match = re.search(rf"\b{ name }\s*\(", prompt)
        if not match:
            continue
        opening = prompt.find("(", match.start())
        end = _invocation_end(prompt, opening)
        if end is None:
            continue
        try:
            call = ast.parse(prompt[match.start():end + 1], mode="eval").body
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != name:
                continue
            if call.args:
                continue
            arguments = {item.arg: ast.literal_eval(item.value) for item in call.keywords if item.arg}
            if len(arguments) != len(call.keywords):
                continue
            return name, arguments
        except (SyntaxError, ValueError, TypeError):
            continue
    return None


def _proposed_terminal_command(text: str, tool_name: str = "terminal_run") -> str | None:
    """Read a terminal tool snippet from the model's response."""
    match = re.search(rf"\b{re.escape(tool_name)}\s*\(", text)
    if not match:
        return None
    opening = text.find("(", match.start())
    end = _invocation_end(text, opening)
    if end is None:
        return None
    body = text[opening + 1:end].strip()
    if not body:
        return None
    try:
        parsed = ast.parse(f"_call({body})", mode="eval").body
        if isinstance(parsed, ast.Call):
            for item in parsed.keywords:
                if item.arg == "command":
                    value = ast.literal_eval(item.value)
                    return value if isinstance(value, str) else None
    except (SyntaxError, ValueError, TypeError):
        pass
    # The local model sometimes prints pseudo-call syntax with raw shell text:
    # terminal_run(curl -X POST ...). Execution is gated by the user's run/API request.
    return body


@dataclass
class AgentResult:
    response: str
    turns: int
    tool_calls: int


class Agent:
    def __init__(self, provider: ChatProvider, tools: ToolRegistry, context: ContextManager, max_turns: int = 40, on_event: Callable[[str, dict], None] | None = None):
        self.provider, self.tools, self.context = provider, tools, context
        self.max_turns = max_turns
        self.on_event = on_event or (lambda *_: None)

    async def _ask_model(self):
        last_error = None
        for attempt in range(3):
            try:
                stream = getattr(self.provider, "chat_stream", None)
                if stream:
                    return await stream(self.context.messages, self.tools.schemas(), lambda token: self.on_event("token", {"text": token}))
                return await self.provider.chat(self.context.messages, self.tools.schemas())
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep((0.4 * (2 ** attempt)) + random.random() * 0.15)
        raise last_error  # type: ignore[misc]

    async def _dispatch_immediately(self, name: str, arguments: dict) -> AgentResult:
        """Execute an explicit read-tool request or user-confirmed command."""
        call_id = f"direct-{len(self.context.messages)}"
        safe_arguments = dict(arguments)
        if isinstance(safe_arguments.get("command"), str):
            safe_arguments["command"] = _redact_sensitive_text(safe_arguments["command"])
        call = ToolCall(id=call_id, name=name, arguments=safe_arguments)
        self.context.add(Message(role="assistant", tool_calls=[call]))
        self.on_event("tool_start", {"name": name, "id": call_id, "arguments": safe_arguments})
        result = await self.tools.dispatch(name, arguments)
        succeeded = json.loads(result).get("ok", False)
        self.on_event("tool_result", {"name": name, "result": result, "id": call_id})
        self.on_event("tool_execution", {"name": name, "status": "succeeded" if succeeded else "failed", "id": call_id})
        self.context.add(Message(role="tool", tool_call_id=call_id, tool_name=name, content=result))
        self.context.save()
        return AgentResult("", 1, 1)

    async def run(self, prompt: str) -> AgentResult:
        curl_command = _curl_login_command(prompt)
        if curl_command:
            self.context.add(Message(role="user", content=_redact_sensitive_text(prompt)))
            return await self._dispatch_immediately(
                "terminal_open", {"command": curl_command, "cwd": "."}
            )
        self.context.add(Message(role="user", content=prompt))
        explicit_read_call = _explicit_read_invocation(prompt)
        if explicit_read_call:
            return await self._dispatch_immediately(*explicit_read_call)

        if _RUN_CONFIRMATION.fullmatch(prompt):
            previous_assistant = next(
                (message.content for message in reversed(self.context.messages[:-1]) if message.role == "assistant"),
                "",
            )
            confirmed_command = _proposed_terminal_command(previous_assistant)
            if confirmed_command:
                return await self._dispatch_immediately(
                    "terminal_open", {"command": confirmed_command, "cwd": "."}
                )

        total_calls = executed_calls = 0
        requires_execution = bool(
            _ACTION_INTENT.search(prompt) or _EXPLICIT_CHANGE_INTENT.search(prompt)
        )
        run_request = bool(_RUN_INTENT.search(prompt))
        curl_request = bool(_CURL_REQUEST_INTENT.search(prompt))
        open_terminal_request = bool(
            _VISIBLE_TERMINAL_INTENT.search(prompt)
            or _RUN_FILE_INTENT.search(prompt)
            or curl_request
        )
        inspection_request = bool(_INSPECT_INTENT.search(prompt))
        process_list_request = bool(_PROCESS_LIST_INTENT.search(prompt))
        process_stop_request = bool(_PROCESS_STOP_INTENT.search(prompt))
        path_request = inspection_request and bool(_PATH_INTENT.search(prompt))
        failed_calls: set[str] = set()
        had_tool_failure = False
        action_retry_count = 0
        requires_tool = requires_execution or run_request or inspection_request or process_list_request or process_stop_request or curl_request

        for turn in range(1, self.max_turns + 1):
            self.context.compact()
            try:
                # Send the registered tool schemas on every model request.
                # The authorization check below still prevents unrequested edits.
                response = await self._ask_model()
            except Exception as exc:
                self.on_event("error", {"message": str(exc)})
                return AgentResult(f"Model request failed: {exc}", turn, total_calls)

            msg = response.message
            self.context.add(msg)
            self.on_event("assistant", {"content": msg.content, "turn": turn})
            log.info("assistant turn=%d tools=%d", turn, len(msg.tool_calls))

            # Treat the prompt as the authorization boundary. A model tool call
            # cannot turn a greeting or question into permission to edit files.
            unauthorized_calls = msg.tool_calls and (
                (not requires_tool)
                or (inspection_request and not requires_execution and not run_request and any(call.name not in {"fs_list", "terminal_run"} for call in msg.tool_calls))
                or (run_request and not requires_execution and any(
                    call.name not in ({"terminal_run", "terminal_open", "project_test"} if open_terminal_request else {"terminal_run"})
                    for call in msg.tool_calls
                ))
                or (process_list_request and not process_stop_request and any(call.name != "process_list" for call in msg.tool_calls))
                or (process_stop_request and any(call.name not in {"process_list", "process_stop", "terminal_stop"} for call in msg.tool_calls))
                or (curl_request and any(call.name not in {"terminal_run", "terminal_open"} for call in msg.tool_calls))
            )
            if unauthorized_calls:
                # Do not persist an unanswered assistant tool call; the next
                # provider request would otherwise contain an orphaned call.
                self.context.messages[-1] = Message(
                    role="assistant",
                    content=msg.content or "How can I help?",
                )
                self.context.save()
                return AgentResult(msg.content or "How can I help?", turn, total_calls)

            if not msg.tool_calls:
                if curl_request or run_request:
                    command = (
                        _proposed_terminal_command(msg.content, "terminal_open")
                        or _proposed_terminal_command(msg.content)
                    )
                    if command:
                        tool_name = "terminal_open" if open_terminal_request else "terminal_run"
                        # Replace the model's prose proposal with an actual
                        # tool-call record before dispatching the command.
                        self.context.messages.pop()
                        return await self._dispatch_immediately(
                            tool_name, {"command": command, "cwd": "."}
                        )
                if requires_tool and executed_calls == 0 and action_retry_count < 1:
                    action_retry_count += 1
                    instruction = (
                        "No workspace change has been made yet. Do not describe a plan: call the appropriate "
                        "filesystem tool now (fs_mkdir, fs_write, fs_edit, or fs_delete)."
                        if requires_execution else
                        "The user asked to run something. Call terminal_open now using the relevant file or command from the conversation."
                        if run_request and open_terminal_request else
                        "The user asked to run something. Call terminal_run now using the relevant file or command from the conversation."
                        if run_request else
                        "The user explicitly requested a curl/API request. Build the curl command from the endpoint and data they supplied, then call terminal_open now so it runs in a visible terminal. Do not ask for confirmation again."
                        if curl_request else
                        "The user asked which processes are running in this workspace. Call process_list now."
                        if process_list_request else
                        "The user asked to stop a process. Identify it with process_list if needed, then call process_stop or terminal_stop with its PID."
                        if process_stop_request else
                        "The user asked for a path. Call terminal_run now with a command that prints the absolute path of the requested file or current workspace."
                        if path_request else
                        "The user asked to inspect workspace contents. Call fs_list with the requested folder path now."
                    )
                    self.context.add(Message(role="user", content=instruction))
                    continue
                if requires_execution and had_tool_failure and action_retry_count < 2:
                    action_retry_count += 1
                    instruction = (
                        "The previous filesystem tool call failed, so the requested change is still incomplete. "
                        "Use the tool result to correct the problem and call a filesystem tool now. "
                        "If replacing an entire file, use fs_write with the complete intended contents; use fs_edit "
                        "only when the exact old text is known. Do not claim success until a tool result confirms it."
                    )
                    self.context.add(Message(role="user", content=instruction))
                    continue
                self.context.save()
                if had_tool_failure:
                    message = "The last tool call failed, so I cannot confirm the work is complete. See the tool error above."
                    self.on_event("error", {"message": message})
                    return AgentResult(message, turn, total_calls)
                if requires_tool and executed_calls == 0:
                    message = (
                        "No filesystem tool was executed, so no workspace changes were made."
                        if requires_execution else
                        "No terminal command was executed, so the requested run was not performed."
                        if run_request else
                        "No terminal command was executed, so the requested curl/API request was not sent."
                        if curl_request else
                        "No process listing tool was executed, so I could not inspect running workspace processes."
                        if process_list_request else
                        "No process stop tool was executed, so no process was stopped."
                        if process_stop_request else
                        "I could not inspect the workspace because no listing tool was executed."
                    )
                    return AgentResult(message, turn, total_calls)
                return AgentResult(msg.content, turn, total_calls)

            total_calls += len(msg.tool_calls)
            failed = False
            for call in msg.tool_calls:
                # Small local models often choose terminal_run even when the
                # user requested a visible terminal. Route the same command to
                # the visible-terminal tool so the request is actually honored.
                tool_name = call.name
                if open_terminal_request and call.name in {"terminal_run", "project_test"}:
                    tool_name = "terminal_open"
                fingerprint = json.dumps([call.name, call.arguments], sort_keys=True, default=str)
                self.on_event("tool_start", {"name": tool_name, "id": call.id, "arguments": call.arguments})
                if failed:
                    result = json.dumps({"ok": False, "error": "Skipped because an earlier tool call failed"})
                elif fingerprint in failed_calls:
                    result = json.dumps({"ok": False, "error": "Repeated identical failed tool call; choose a different strategy"})
                    failed = True
                else:
                    result = await self.tools.dispatch(tool_name, call.arguments)
                    executed_calls += 1
                    failed = not json.loads(result).get("ok", False)
                    if failed:
                        failed_calls.add(fingerprint)
                self.on_event("tool_result", {"name": tool_name, "result": result, "id": call.id})
                self.on_event("tool_execution", {"name": tool_name, "status": "failed" if failed else "succeeded", "id": call.id})
                self.context.add(Message(role="tool", tool_call_id=call.id, tool_name=tool_name, content=result))
                self.context.save()
            had_tool_failure = failed

        self.context.save()
        return AgentResult("Stopped after reaching the maximum number of reasoning turns.", self.max_turns, total_calls)
