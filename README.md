# Local AI Agent Harness (workspace and terminal tools)

A small, extensible Python harness for local models. The Ollama model gets workspace file tools and terminal access to build, run, and serve projects inside the current folder.

The model can run shell commands in the workspace and start detached long-running processes such as a web server.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cd C:\path\to\your\project
agent
```

Type your task at `You>` (for example `Create a todo app with a Flask backend and an HTML/JS frontend`). Pasted text stays in the prompt as a draft; add or edit text as needed, then press Enter to submit the complete message, including pasted line breaks. `/exit` ends the session. You can also run a single task: `agent "create a portfolio website"`.

To use `agent` in any terminal without activating a venv, run `pip install -e .` once with your global Python (or use `pipx install -e .`).

## Generate and host a page

Run the agent from the folder where you want the page files, then ask it to create and serve a page:

```powershell
agent "Create a simple static website in site/index.html, then serve the site folder on port 8000 and tell me the URL and PID."
```

The agent can use `terminal_start` to run `python -m http.server 8000` from `site`, then open `http://localhost:8000` in a browser. The server runs independently of the agent session; ask the agent to stop it using the PID it returned, or stop it manually from your operating system.

On startup the harness reads `.env` from the current folder, falling back to the harness installation folder when the current folder has no `.env`. Ollama remains the default provider; it starts Ollama if needed and pulls the selected model if it is missing. Choose `huggingface` or `groq` with `MODEL_PROVIDER` to use their hosted chat APIs instead.

```env
MODEL_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:3b
# OLLAMA_HOST=http://127.0.0.1:11434

# Hugging Face Inference Providers
# MODEL_PROVIDER=huggingface
# HF_MODEL=provider/model-name
# HF_TOKEN=your-hugging-face-access-token

# Groq
# MODEL_PROVIDER=groq
# GROQ_MODEL=openai/gpt-oss-20b
# GROQ_API_KEY=your-groq-api-key
```

`MODEL_NAME` can select a model for any provider. `HF_MODEL` and `GROQ_MODEL` are provider-specific alternatives. Set `HF_TOKEN` or `GROQ_API_KEY` for the matching hosted provider. The application reads these values directly from `.env`; process environment variables are not used. Provider selection is centralized in `models/provider.py`. Free usage, model availability, and rate limits depend on the provider's current account plan; a model ID alone does not guarantee free access.

The Hugging Face adapter uses `https://router.huggingface.co/v1/chat/completions`; the Groq adapter uses `https://api.groq.com/openai/v1/chat/completions`. Both support the harness's standard chat and function-tool request format.

## Tools

| Tool | What it does |
| --- | --- |
| `fs_list` | list files and folders directly inside a folder (read-only) |
| `fs_read` | read a UTF-8 text file |
| `fs_search` | search file names and text contents under a folder (up to 200 matches) |
| `fs_mkdir` | create a folder (parents included) |
| `fs_write` | create a file or fully replace it (parent folders are created automatically) |
| `fs_edit` | update a file by replacing one exact, unique piece of text |
| `fs_patch` | apply a unified diff to an existing file |
| `fs_rename` | rename or move a file within the workspace (destination must not exist) |
| `fs_delete` | delete one existing file (folders are not deleted) |
| `fs_delete_files` | delete all files directly inside a folder; optionally include subfolders with `recursive=true` (folders remain) |
| `terminal_run` | run a shell command in the workspace and capture output (1-120 second timeout) |
| `project_test` | run a test, lint, or build command (1-1200 second timeout) |
| `terminal_start` | start a process in the background without opening a terminal window; returns its PID |
| `terminal_open` | run a finite command in a visible terminal, return output and exit code, then close it |
| `terminal_stop` | stop a background process started by `terminal_start` |
| `process_list` | list accessible processes whose current working directory is in the workspace |
| `process_stop` | stop a process by PID if its current working directory is in the workspace; also stops in-workspace child processes |
| `git_status` | show branch and working tree status (read-only) |
| `git_diff` | show unstaged changes, optionally for one file (read-only) |

Paths are relative to the workspace (the folder you run `agent` in, or `--workspace`).

## Architecture

- `models` : provider-neutral messages and adapters for Ollama, Hugging Face Inference Providers, and Groq
- `core` : the agent loop and the dynamic tool registry
- `runtime` : owns the workspace folder and resolves paths
- `tools` : filesystem tools
- `context` : persists and compacts sessions (`.harness/sessions`)
- `cli` : wires everything together

Every tool schema is sent to the model, arguments are validated before dispatch, and every result (including failures) is added to the conversation before the next model request. Shell commands run with their working directory inside the selected workspace. Commands can still perform actions available to the current user account.

## Options

`--workspace`, `--sessions`, `--session`, `--host`, `--max-turns` (default 40), `--verbose`.
Environment in `.env`: `MODEL_PROVIDER`, `MODEL_NAME`, `OLLAMA_MODEL`, `HF_MODEL`, `GROQ_MODEL`, provider tokens, `OLLAMA_HOST`, `HARNESS_WORKSPACE`, `HARNESS_SESSION_DIR`, `HARNESS_MAX_TURNS`.

The agent detects whether it was launched from PowerShell, CMD, or Git Bash and uses that shell for terminal tools. On Windows, curl commands always open in Git Bash. Set `HARNESS_SHELL` to `powershell`, `cmd`, or `git-bash` to override detection, or set `HARNESS_GIT_BASH` to the full path of `bash.exe` if Git Bash is not discoverable through Git for Windows' default locations.

Run tests with `pytest`.

## Notes for a 3B model

`qwen2.5:3b` can be unreliable at tool calling. Keep tasks small and specific, and review generated code. If it struggles, try a larger model such as `qwen2.5-coder:7b` via `OLLAMA_MODEL`.
