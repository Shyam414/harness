from __future__ import annotations

from typing import Any, Protocol
from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None


class ChatResponse(BaseModel):
    message: Message
    done: bool = True


class ChatProvider(Protocol):
    async def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ChatResponse: ...


class ProviderError(RuntimeError):
    """Provider request or response failed."""
