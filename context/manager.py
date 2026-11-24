from __future__ import annotations

import json
import os
from pathlib import Path
from models.base import Message


class ContextManager:
    def __init__(self, session_dir: Path, session_id: str = "default", max_messages: int = 80, max_tokens: int = 24_000):
        self.directory = session_dir
        self.directory.mkdir(parents=True, exist_ok=True)
        self._hide_harness_directory()
        self.path = self.directory / f"{session_id}.json"
        self.max_messages, self.max_tokens = max_messages, max_tokens
        self.messages: list[Message] = []
        if self.path.exists():
            try:
                self.messages = [Message.model_validate(x) for x in json.loads(self.path.read_text(encoding="utf-8"))]
            except (ValueError, OSError):
                self.messages = []

    def _hide_harness_directory(self) -> None:
        """Mark the default .harness directory hidden in Windows Explorer."""
        if os.name != "nt":
            return

        harness_dir = next((parent for parent in (self.directory, *self.directory.parents)
                            if parent.name.lower() == ".harness"), None)
        if harness_dir is None:
            return

        import ctypes

        attributes = ctypes.windll.kernel32.GetFileAttributesW(str(harness_dir))
        if attributes == 0xFFFFFFFF:
            return
        ctypes.windll.kernel32.SetFileAttributesW(str(harness_dir), attributes | 0x2)

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def compact(self) -> None:
        if len(self.messages) <= self.max_messages and self._estimated_tokens() <= self.max_tokens:
            return
        # Keep the system instruction and recent history within a conservative character estimate.
        system = [m for m in self.messages if m.role == "system"][:1]
        recent = [m for m in self.messages if m.role != "system"]
        kept = []
        budget = max(256, self.max_tokens * 4 - sum(len(m.content) for m in system))
        used = 0
        for message in reversed(recent):
            cost = len(message.content) + 100
            if kept and (len(kept) + 1 >= self.max_messages or used + cost > budget):
                break
            kept.append(message); used += cost
        self.messages = system + list(reversed(kept))

    def _estimated_tokens(self) -> int:
        # A deliberately conservative approximation that works without a model tokenizer.
        return sum((len(message.content) + 3) // 4 + 16 for message in self.messages)

    def save(self) -> None:
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps([m.model_dump(mode="json") for m in self.messages], ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)
