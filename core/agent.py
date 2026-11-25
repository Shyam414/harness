from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import Callable

from context.manager import ContextManager
from models.base import ChatProvider, Message, ToolCall
from .registry import ToolRegistry
from .intents import (
    ACTION_INTENT,
    CURL_REQUEST_INTENT,
    EXPLICIT_CHANGE_INTENT,
    INSPECT_INTENT,
    PATH_INTENT,
    PROCESS_LIST_INTENT,
    PROCESS_STOP_INTENT,
    RUN_CONFIRMATION,
    RUN_FILE_INTENT,
    RUN_INTENT,
    VISIBLE_TERMINAL_INTENT,
)
from .request_parsing import (
    curl_login_command,
    explicit_read_invocation,
    proposed_terminal_command,
    redact_sensitive_text,
)

log = logging.getLogger("harness.agent")

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
            safe_arguments["command"] = redact_sensitive_text(safe_arguments["command"])
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
        curl_command = curl_login_command(prompt)
        if curl_command:
            self.context.add(Message(role="user", content=redact_sensitive_text(prompt)))
            return await self._dispatch_immediately(
                "terminal_open", {"command": curl_command, "cwd": "."}
            )
        self.context.add(Message(role="user", content=prompt))
        explicit_read_call = explicit_read_invocation(prompt)
        if explicit_read_call:
            return await self._dispatch_immediately(*explicit_read_call)

        if RUN_CONFIRMATION.fullmatch(prompt):
            previous_assistant = next(
                (message.content for message in reversed(self.context.messages[:-1]) if message.role == "assistant"),
                "",
            )
            confirmed_command = proposed_terminal_command(previous_assistant)
            if confirmed_command:
                return await self._dispatch_immediately(
                    "terminal_open", {"command": confirmed_command, "cwd": "."}
                )

        total_calls = executed_calls = 0
        requires_execution = bool(
            ACTION_INTENT.search(prompt) or EXPLICIT_CHANGE_INTENT.search(prompt)
        )
        run_request = bool(RUN_INTENT.search(prompt))
        curl_request = bool(CURL_REQUEST_INTENT.search(prompt))
        open_terminal_request = bool(
            VISIBLE_TERMINAL_INTENT.search(prompt)
            or RUN_FILE_INTENT.search(prompt)
            or curl_request
        )
        inspection_request = bool(INSPECT_INTENT.search(prompt))
        process_list_request = bool(PROCESS_LIST_INTENT.search(prompt))
        process_stop_request = bool(PROCESS_STOP_INTENT.search(prompt))
        path_request = inspection_request and bool(PATH_INTENT.search(prompt))
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
                        proposed_terminal_command(msg.content, "terminal_open")
                        or proposed_terminal_command(msg.content)
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
                        "Read the exact error and correct both the workspace-relative path and tool arguments. "
                        "If the target file exists, use fs_read before editing it. If replacing or repairing a whole "
                        "file, use fs_write with the complete intended contents, excluding Markdown fence lines; use "
                        "fs_edit only when the exact old text is known. Do not send a plan or claim success: call the "
                        "appropriate filesystem tool now, then report completion only after its successful result."
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
