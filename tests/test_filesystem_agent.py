import json
from pathlib import Path

from context.manager import ContextManager
from core.agent import Agent
from core.registry import ToolRegistry
from models.base import ChatResponse, Message, ToolCall
from runtime.manager import RuntimeManager
from tools.filesystem import register_filesystem


class ScriptedProvider:
    def __init__(self, steps):
        self.steps = list(steps)

    async def chat(self, messages, tools):
        return ChatResponse(message=self.steps.pop(0))


def call(i, name, **arguments):
    return ToolCall(id=f"c{i}", name=name, arguments=arguments)


async def test_create_and_update(tmp_path: Path):
    runtime = RuntimeManager(tmp_path / "proj")
    registry = ToolRegistry()
    register_filesystem(registry, runtime)
    ctx = ContextManager(tmp_path / "sessions")
    provider = ScriptedProvider([
        Message(role="assistant", tool_calls=[
            call(1, "fs_mkdir", path="frontend"),
            call(2, "fs_write", path="backend/app.py", content="print('v1')\n"),
            call(3, "fs_write", path="frontend/index.html", content="<h1>Hi</h1>\n"),
        ]),
        Message(role="assistant", tool_calls=[call(4, "fs_edit", path="backend/app.py", old_text="v1", new_text="v2")]),
        Message(role="assistant", content="Created the project."),
    ])
    agent = Agent(provider, registry, ctx)
    result = await agent.run("create a small app")
    assert result.tool_calls == 4 and result.response == "Created the project."
    assert (tmp_path / "proj/frontend/index.html").read_text() == "<h1>Hi</h1>\n"
    assert (tmp_path / "proj/backend/app.py").read_text() == "print('v2')\n"


async def test_failed_edit_is_reported(tmp_path: Path):
    runtime = RuntimeManager(tmp_path)
    registry = ToolRegistry()
    register_filesystem(registry, runtime)
    out = json.loads(await registry.dispatch("fs_edit", {"path": "missing.txt", "old_text": "a", "new_text": "b"}))
    assert out["ok"] is False


async def test_agent_recovers_from_failed_edit_with_full_file_write(tmp_path: Path):
    runtime = RuntimeManager(tmp_path / "proj")
    registry = ToolRegistry()
    register_filesystem(registry, runtime)
    ctx = ContextManager(tmp_path / "sessions")
    provider = ScriptedProvider([
        Message(role="assistant", tool_calls=[
            call(1, "fs_edit", path="hello.py", old_text="missing text", new_text="replacement")
        ]),
        Message(role="assistant", content="I will try again."),
        Message(role="assistant", tool_calls=[
            call(2, "fs_write", path="hello.py", content="from flask import Flask\n\napp = Flask(__name__)\n")
        ]),
        Message(role="assistant", content="Updated hello.py.")
    ])
    agent = Agent(provider, registry, ctx)

    result = await agent.run("update hello.py with a Flask app")

    assert result.response == "Updated hello.py."
    assert result.tool_calls == 2
    assert (tmp_path / "proj" / "hello.py").read_text() == "from flask import Flask\n\napp = Flask(__name__)\n"
