from __future__ import annotations

from pydantic import BaseModel, Field

from core.registry import Tool, ToolRegistry
from runtime.manager import RuntimeManager


class RunArgs(BaseModel):
    command: str
    cwd: str = "."
    timeout_seconds: int = Field(default=30, ge=1, le=120)


class StartArgs(BaseModel):
    command: str
    cwd: str = "."


class OpenArgs(BaseModel):
    command: str
    cwd: str = "."
    timeout_seconds: int = Field(default=120, ge=1, le=1200)


class StopArgs(BaseModel):
    pid: int = Field(gt=0)


class ProcessListArgs(BaseModel):
    pass


class ProcessStopArgs(BaseModel):
    pid: int = Field(gt=0)


class ProjectTestArgs(BaseModel):
    command: str
    cwd: str = "."
    timeout_seconds: int = Field(default=120, ge=1, le=1200)


def register_terminal(registry: ToolRegistry, runtime: RuntimeManager) -> None:
    """Register workspace-scoped shell execution and detached process controls."""
    registry.register(Tool(
        "terminal_run",
        "Run a shell command in the workspace (or a relative subfolder), wait for it, and return stdout, stderr, and exit code. Timeout is 1-120 seconds.",
        RunArgs,
        runtime.run_command,
    ))
    registry.register(Tool(
        "terminal_start",
        "Start a long-running command in the background without opening a terminal window, for example a local web server. Returns a PID. Use terminal_stop with that PID when finished.",
        StartArgs,
        runtime.start_command,
    ))
    registry.register(Tool(
        "terminal_open",
        "Open a visible terminal in the workspace and run a finite command there. The agent waits for completion, captures the displayed output and exit code, then the terminal closes. On Windows, curl commands use Git Bash only. Timeout is 1-1200 seconds. Use terminal_start for a process that should remain running.",
        OpenArgs,
        runtime.open_terminal,
    ))
    registry.register(Tool(
        "terminal_stop",
        "Stop a background process started by terminal_start in this agent session. Requires the PID returned by that tool.",
        StopArgs,
        runtime.stop_command,
    ))
    registry.register(Tool(
        "process_list",
        "List accessible running processes whose current working directory is inside this workspace. Returns PID, name, command, and working directory. The agent and its parent shells are excluded.",
        ProcessListArgs,
        runtime.list_workspace_processes,
    ))
    registry.register(Tool(
        "process_stop",
        "Stop a running process by PID only if its current working directory is inside this workspace. Also terminates its child processes. Refuses to stop the agent or its parent shells. Use process_list first to identify the intended process.",
        ProcessStopArgs,
        runtime.stop_workspace_process,
    ))
    registry.register(Tool(
        "project_test",
        "Run a project test, lint, or build command in the workspace and return output and exit code. Timeout is 1-1200 seconds.",
        ProjectTestArgs,
        runtime.run_command,
    ))
