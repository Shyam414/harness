from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable
from pydantic import BaseModel, ValidationError

log = logging.getLogger("tools")
ToolHandler = Callable[..., Any | Awaitable[Any]]


class Tool:
    def __init__(self, name: str, description: str, args_model: type[BaseModel], handler: ToolHandler):
        self.name, self.description, self.args_model, self.handler = name, description, args_model, handler

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.args_model.model_json_schema()}}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"ok": False, "error": f"Unknown tool: {name}"})
        try:
            validated = tool.args_model.model_validate(arguments)
            values = validated.model_dump()
            result = tool.handler(**values)
            if asyncio.iscoroutine(result):
                result = await result
            log.info("tool=%s completed", name)
            return json.dumps({"ok": True, "result": result}, ensure_ascii=False, default=str)
        except (ValidationError, ValueError, OSError, RuntimeError) as exc:
            log.warning("tool=%s failed: %s", name, type(exc).__name__)
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        except Exception as exc:
            log.error("tool=%s unexpected failure: %s", name, type(exc).__name__)
            return json.dumps({"ok": False, "error": f"Tool failed: {type(exc).__name__}"}, ensure_ascii=False)
