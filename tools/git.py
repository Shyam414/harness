from __future__ import annotations

import subprocess

from pydantic import BaseModel

from core.registry import Tool, ToolRegistry
from runtime.manager import RuntimeManager


class GitPathArgs(BaseModel):
    path: str = "."


class GitDiffArgs(BaseModel):
    path: str = "."
    file_path: str | None = None


def register_git(registry: ToolRegistry, runtime: RuntimeManager) -> None:
    """Register read-only Git inspection tools."""

    def git_status(path: str = "."):
        directory = runtime.resolve(path)
        if not directory.is_dir():
            raise ValueError(f"Not an existing folder: {path}")
        result = subprocess.run(["git", "status", "--short", "--branch"], cwd=directory, capture_output=True, text=True, timeout=15, check=False)
        return {"path": path, "returncode": result.returncode, "stdout": result.stdout[-20000:], "stderr": result.stderr[-20000:]}

    def git_diff(path: str = ".", file_path: str | None = None):
        directory = runtime.resolve(path)
        if not directory.is_dir():
            raise ValueError(f"Not an existing folder: {path}")
        args = ["git", "diff", "--"]
        if file_path:
            target = runtime.resolve(file_path)
            try:
                relative = target.relative_to(directory).as_posix()
            except ValueError as exc:
                raise ValueError("file_path must be inside the selected Git folder") from exc
            args.append(relative)
        result = subprocess.run(args, cwd=directory, capture_output=True, text=True, timeout=15, check=False)
        return {"path": path, "file_path": file_path, "returncode": result.returncode, "stdout": result.stdout[-30000:], "stderr": result.stderr[-20000:]}

    registry.register(Tool("git_status", "Show the Git branch and working tree changes in a workspace folder. Read-only.", GitPathArgs, git_status))
    registry.register(Tool("git_diff", "Show the unstaged Git diff in a workspace folder, optionally limited to one workspace file. Read-only.", GitDiffArgs, git_diff))
