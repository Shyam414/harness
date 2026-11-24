from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import psutil


_VISIBLE_CAPTURE_HELPER = r'''
import json, os, subprocess, sys
argv = json.loads(sys.argv[1])
output_path, status_path = sys.argv[2], sys.argv[3]
tail, total = bytearray(), 0
try:
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        chunk = os.read(process.stdout.fileno(), 4096)
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        total += len(chunk)
        tail.extend(chunk)
        if len(tail) > 20000:
            del tail[:-20000]
        with open(output_path, "wb") as output:
            output.write(tail)
    returncode = process.wait()
    with open(status_path, "w", encoding="utf-8") as status:
        json.dump({"returncode": returncode, "truncated": total > len(tail)}, status)
except Exception as error:
    message = (str(error) + "\n").encode(errors="replace")
    sys.stderr.buffer.write(message)
    sys.stderr.buffer.flush()
    with open(output_path, "wb") as output:
        output.write(message[-20000:])
    with open(status_path, "w", encoding="utf-8") as status:
        json.dump({"returncode": 127, "error": str(error), "truncated": False}, status)
'''


class RuntimeManager:
    """Owns the working directory and processes started by workspace tools."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._processes: dict[int, subprocess.Popen] = {}
        self.shell_name, self.shell_executable = self._detect_shell()

    def _detect_shell(self) -> tuple[str, str]:
        """Select the shell inherited from the agent's launch terminal."""
        requested = os.getenv("HARNESS_SHELL", "").strip().lower()
        aliases = {
            "cmd": "cmd", "cmd.exe": "cmd",
            "powershell": "powershell", "powershell.exe": "powershell", "pwsh": "powershell",
            "bash": "git-bash", "gitbash": "git-bash", "git-bash": "git-bash",
        }
        if requested and requested not in aliases:
            raise ValueError("HARNESS_SHELL must be cmd, powershell, or git-bash")
        selected = aliases.get(requested)

        if os.name == "nt" and selected is None:
            # Prefer the actual parent terminal over environment guesses. This
            # distinguishes PowerShell and Git Bash even when COMSPEC is cmd.exe.
            try:
                for parent in psutil.Process().parents()[:8]:
                    parent_name = parent.name().lower()
                    if parent_name in {"powershell.exe", "pwsh.exe"}:
                        selected = "powershell"
                        break
                    if parent_name in {"bash.exe", "git-bash.exe"}:
                        selected = "git-bash"
                        break
                    if parent_name == "cmd.exe":
                        selected = "cmd"
                        break
            except psutil.Error:
                pass

        if os.name == "nt":
            selected = selected or "cmd"
            if selected == "cmd":
                return "CMD", os.getenv("COMSPEC", "cmd.exe")
            if selected == "powershell":
                return "PowerShell", shutil.which("pwsh.exe") or shutil.which("powershell.exe") or "powershell.exe"
            bash = self._git_bash_executable()
            if not bash:
                raise ValueError("Git Bash was selected, but bash.exe was not found on PATH")
            return "Git Bash", bash

        shell = os.getenv("SHELL") or "/bin/sh"
        return Path(shell).name, shell

    @staticmethod
    def _git_bash_executable() -> str | None:
        """Find Git for Windows' Bash without mistaking WSL's bash.exe for it."""
        explicit = os.getenv("HARNESS_GIT_BASH")
        candidates = [Path(explicit)] if explicit else []
        git_executable = shutil.which("git.exe") or shutil.which("git")
        if git_executable:
            candidates.append(Path(git_executable).resolve().parent.parent / "bin" / "bash.exe")
        for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            root = os.getenv(variable)
            if root:
                base = Path(root)
                candidates.extend((base / "Git" / "bin" / "bash.exe", base / "Programs" / "Git" / "bin" / "bash.exe"))
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
        located = shutil.which("bash.exe") or shutil.which("bash")
        if located and "git" in str(Path(located)).lower():
            return located
        return None

    @staticmethod
    def _is_curl_command(command: str) -> bool:
        return bool(re.match(r"\s*curl(?:\.exe)?(?:\s|$)", command, re.IGNORECASE))

    def shell_info(self) -> dict[str, str]:
        """Describe the actual shell used by terminal_run and terminal_open."""
        info = {"name": self.shell_name, "executable": self.shell_executable}
        if os.name == "nt":
            info["curl_shell"] = "Git Bash (required)"
        return info

    def _command_args(self, command: str, shell_name: str | None = None, executable: str | None = None) -> list[str]:
        shell_name = shell_name or self.shell_name
        executable = executable or self.shell_executable
        if shell_name == "CMD":
            return [executable, "/d", "/s", "/c", command]
        if shell_name == "PowerShell":
            return [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command]
        return [executable, "-lc", command]

    def resolve(self, path: str) -> Path:
        """Turn a tool-supplied path into a filesystem path inside the workspace folder."""
        target = (self.workspace / str(path).strip().replace("\\", "/")).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError("Path must stay inside the workspace") from exc
        return target

    @staticmethod
    def _windows_curl_command(command: str) -> str:
        """Convert common POSIX single-quoted JSON curl data to CMD quoting."""
        if not re.match(r"\s*curl(?:\.exe)?\s", command, re.IGNORECASE):
            return command
        pattern = re.compile(r"(?P<option>(?:-d|--data|--data-raw)\s+)'(?P<body>\{.*?\})'(?=\s|$)", re.DOTALL)

        def replace(match: re.Match[str]) -> str:
            try:
                payload = json.loads(match.group("body"))
            except (json.JSONDecodeError, TypeError):
                return match.group(0)
            compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            # CMD passes these backslash-escaped quotes to curl.exe as JSON.
            escaped = compact.replace('"', '\\"')
            return f'{match.group("option")}"{escaped}"'

        command = re.sub(r"\\\r?\n", " ", command)
        return pattern.sub(replace, command)

    @staticmethod
    def _powershell_curl_command(command: str) -> str:
        """Bypass PowerShell's curl alias when the command uses curl CLI flags."""
        return re.sub(r"^(\s*)curl(?=\s|$)", r"\1curl.exe", command, count=1, flags=re.IGNORECASE)

    async def run_command(self, command: str, cwd: str = ".", timeout_seconds: int = 30) -> dict:
        """Run a shell command in a workspace directory and capture bounded output."""
        directory = self.resolve(cwd)
        if not directory.is_dir():
            raise ValueError(f"Working directory is not a folder: {cwd}")
        if not command.strip():
            raise ValueError("Command must not be empty")
        if os.name == "nt" and self._is_curl_command(command):
            raise RuntimeError("On Windows, curl commands must use terminal_open so they run visibly in Git Bash.")
        if self.shell_name == "CMD":
            command = self._windows_curl_command(command)

        process = await asyncio.create_subprocess_exec(
            *self._command_args(command),
            cwd=directory,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
            return {
                "cwd": str(directory),
                "returncode": process.returncode,
                "stdout": stdout.decode(errors="replace")[-20_000:],
                "stderr": stderr.decode(errors="replace")[-20_000:],
                "truncated": len(stdout) > 20_000 or len(stderr) > 20_000,
            }
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {"cwd": str(directory), "returncode": None, "timed_out": True, "timeout_seconds": timeout_seconds}

    def start_command(self, command: str, cwd: str = ".") -> dict:
        """Start a detached command whose lifetime is independent of the agent session."""
        directory = self.resolve(cwd)
        if not directory.is_dir():
            raise ValueError(f"Working directory is not a folder: {cwd}")
        if not command.strip():
            raise ValueError("Command must not be empty")
        if os.name == "nt" and self._is_curl_command(command):
            raise ValueError("On Windows, curl commands must use terminal_open so they run visibly in Git Bash.")
        kwargs = {
            "args": self._command_args(command),
            "cwd": directory,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(**kwargs)
        self._processes[process.pid] = process
        return {"pid": process.pid, "cwd": str(directory), "started": True, "shell": self.shell_name}

    async def open_terminal(self, command: str, cwd: str = ".", timeout_seconds: int = 120) -> dict:
        """Run visibly, capture its output for the agent, and close the window on completion."""
        directory = self.resolve(cwd)
        if not directory.is_dir():
            raise ValueError(f"Working directory is not a folder: {cwd}")
        if not command.strip():
            raise ValueError("Command must not be empty")

        execution_shell, execution_executable = self.shell_name, self.shell_executable
        if os.name == "nt" and self._is_curl_command(command):
            execution_executable = self._git_bash_executable()
            if not execution_executable:
                raise RuntimeError("Curl commands require Git Bash, but Git Bash was not found. Install Git for Windows or set HARNESS_GIT_BASH to bash.exe.")
            execution_shell = "Git Bash"

        if execution_shell == "CMD":
            command = self._windows_curl_command(command)
        elif execution_shell == "PowerShell":
            command = self._powershell_curl_command(command)

        argv = self._command_args(command, execution_shell, execution_executable)
        output_file = tempfile.NamedTemporaryFile(prefix="harness-terminal-", suffix=".log", delete=False)
        status_file = tempfile.NamedTemporaryFile(prefix="harness-terminal-", suffix=".json", delete=False)
        output_path, status_path = output_file.name, status_file.name
        output_file.close()
        status_file.close()
        wrapper_args = [sys.executable, "-u", "-c", _VISIBLE_CAPTURE_HELPER, json.dumps(argv), output_path, status_path]

        try:
            if os.name == "nt":
                process = subprocess.Popen(
                    wrapper_args,
                    cwd=directory,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
            else:
                terminal = next(
                    (shutil.which(name) for name in ("x-terminal-emulator", "gnome-terminal", "konsole", "xterm") if shutil.which(name)),
                    None,
                )
                if terminal is None:
                    raise RuntimeError("No supported terminal emulator found (x-terminal-emulator, gnome-terminal, konsole, or xterm)")
                name = os.path.basename(terminal)
                if name == "gnome-terminal":
                    args = [terminal, "--"] + wrapper_args
                else:
                    args = [terminal, "-e"] + wrapper_args
                process = subprocess.Popen(args, cwd=directory, start_new_session=True)

            timed_out = False
            deadline = time.monotonic() + timeout_seconds
            if os.name == "nt":
                try:
                    await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    timed_out = True
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
                    await asyncio.to_thread(process.wait)
            else:
                status = None
                while status is None and time.monotonic() < deadline:
                    try:
                        status = json.loads(Path(status_path).read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        pass
                    if status is None:
                        await asyncio.sleep(0.1)
                timed_out = status is None
                if timed_out:
                    process.terminate()
            if os.name == "nt":
                status = json.loads(Path(status_path).read_text(encoding="utf-8")) if Path(status_path).exists() else {}

            output = Path(output_path).read_bytes()[-20_000:].decode(errors="replace") if Path(output_path).exists() else ""
            status = status or {}
            return {
                "pid": process.pid,
                "cwd": str(directory),
                "returncode": status.get("returncode"),
                "stdout": output,
                "stderr": "",
                "timed_out": timed_out,
                "truncated": status.get("truncated", False),
                "terminal_closed": not timed_out,
                "terminal_opened": True,
                "shell": execution_shell,
            }
        finally:
            for path in (output_path, status_path):
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass

    def _protected_process_ids(self) -> set[int]:
        """Protect this agent and its ancestor shells from process-stop requests."""
        protected = {os.getpid()}
        try:
            protected.update(parent.pid for parent in psutil.Process().parents())
        except psutil.Error:
            pass
        return protected

    def _process_workspace_cwd(self, process: psutil.Process) -> Path | None:
        try:
            cwd = process.cwd()
            if not cwd:
                return None
            path = Path(cwd).resolve()
            path.relative_to(self.workspace)
            return path
        except (psutil.Error, OSError, ValueError):
            return None

    def list_workspace_processes(self) -> dict:
        """List accessible processes whose current working directory is in this workspace."""
        protected = self._protected_process_ids()
        processes = []
        for process in psutil.process_iter(["pid", "name", "cmdline", "status"]):
            if process.info["pid"] in protected:
                continue
            cwd = self._process_workspace_cwd(process)
            if cwd is None:
                continue
            command = process.info.get("cmdline") or []
            processes.append({
                "pid": process.info["pid"],
                "name": process.info.get("name") or "unknown",
                "cwd": str(cwd),
                "command": " ".join(command)[:500],
                "status": process.info.get("status"),
            })
        processes.sort(key=lambda item: item["pid"])
        return {"workspace": str(self.workspace), "processes": processes}

    def stop_workspace_process(self, pid: int) -> dict:
        """Terminate a process only when its current working directory is in this workspace."""
        if pid in self._protected_process_ids():
            raise ValueError("Refusing to stop the agent process or one of its parent shells")
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess as exc:
            raise ValueError(f"No running process with PID {pid}") from exc
        cwd = self._process_workspace_cwd(process)
        if cwd is None:
            raise ValueError("Process is inaccessible or its current directory is outside this workspace")

        try:
            children = process.children(recursive=True)
        except psutil.Error as exc:
            raise ValueError(f"Cannot inspect child processes for PID {pid}: {exc}") from exc
        # Keep the operation scoped to this workspace even if the process has
        # spawned children that changed their working directory elsewhere.
        targets = [child for child in children if self._process_workspace_cwd(child) is not None] + [process]
        targets = list({child.pid: child for child in targets}.values())
        for target in targets:
            try:
                target.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(targets, timeout=3)
        for target in alive:
            try:
                target.kill()
            except psutil.NoSuchProcess:
                pass
        return {"pid": pid, "cwd": str(cwd), "stopped_pids": [target.pid for target in targets], "stopped": True}

    def stop_command(self, pid: int) -> dict:
        """Stop a process started by this runtime."""
        process = self._processes.get(pid)
        if process is None:
            raise ValueError("PID was not started by this agent session")
        if process.poll() is not None:
            self._processes.pop(pid, None)
            return {"pid": pid, "stopped": False, "status": "already exited"}
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        self._processes.pop(pid, None)
        return {"pid": pid, "stopped": True}

    async def close(self) -> None:
        return None
