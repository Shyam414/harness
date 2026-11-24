from __future__ import annotations

import re
from pydantic import BaseModel
from core.registry import Tool, ToolRegistry
from runtime.manager import RuntimeManager


class WriteArgs(BaseModel):
    path: str
    content: str


class EditArgs(BaseModel):
    path: str
    old_text: str
    new_text: str


class MkdirArgs(BaseModel):
    path: str


class DeleteArgs(BaseModel):
    path: str


class RenameArgs(BaseModel):
    path: str
    new_path: str


class DeleteFilesArgs(BaseModel):
    path: str
    recursive: bool = False


class ListArgs(BaseModel):
    path: str = "."


class ReadArgs(BaseModel):
    path: str


class SearchArgs(BaseModel):
    query: str
    path: str = "."


class PatchArgs(BaseModel):
    path: str
    patch: str


def register_filesystem(registry: ToolRegistry, runtime: RuntimeManager) -> None:
    """Register workspace file and folder tools."""

    def mkdir(path: str):
        target = runtime.resolve(path)
        existed = target.is_dir()
        target.mkdir(parents=True, exist_ok=True)
        return {"path": path, "created": not existed}

    def write(path: str, content: str):
        target = runtime.resolve(path)
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        return {"path": path, "bytes": target.stat().st_size, "created": not existed}

    def edit(path: str, old_text: str, new_text: str):
        target = runtime.resolve(path)
        content = target.read_text(encoding="utf-8")
        if old_text not in content:
            raise ValueError("Text to replace was not found")
        if content.count(old_text) != 1:
            raise ValueError("Text to replace must occur exactly once")
        target.write_text(content.replace(old_text, new_text), encoding="utf-8", newline="\n")
        return {"path": path, "edited": True}

    def read(path: str):
        target = runtime.resolve(path)
        if not target.is_file():
            raise ValueError(f"Not an existing file: {path}")
        return {"path": path, "content": target.read_text(encoding="utf-8")}

    def search(query: str, path: str = "."):
        if not query:
            raise ValueError("Search query must not be empty")
        root = runtime.resolve(path)
        if not root.is_dir():
            raise ValueError(f"Not an existing folder: {path}")
        ignored = {".git", ".venv", "venv", "__pycache__", ".harness"}
        hits = []
        needle = query.casefold()
        for item in root.rglob("*"):
            if not item.is_file() or any(part in ignored for part in item.relative_to(root).parts):
                continue
            try:
                item.resolve().relative_to(runtime.workspace)
            except ValueError:
                continue
            rel = item.relative_to(runtime.workspace).as_posix()
            if needle in item.name.casefold():
                hits.append({"path": rel, "match": "filename"})
            try:
                for number, line in enumerate(item.read_text(encoding="utf-8").splitlines(), 1):
                    if needle in line.casefold():
                        hits.append({"path": rel, "line": number, "text": line[:500]})
                        if len(hits) >= 200:
                            return {"query": query, "hits": hits, "truncated": True}
            except (UnicodeDecodeError, OSError):
                continue
            if len(hits) >= 200:
                return {"query": query, "hits": hits, "truncated": True}
        return {"query": query, "hits": hits, "truncated": False}

    def apply_patch(path: str, patch: str):
        target = runtime.resolve(path)
        if not target.is_file():
            raise ValueError(f"Not an existing file: {path}")
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        hunks = []
        current = None
        for line in patch.splitlines(keepends=True):
            if line.startswith("@@"):
                match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                if not match:
                    raise ValueError("Invalid unified diff hunk header")
                current = {"start": int(match.group(1)) - 1, "old": [], "new": []}
                hunks.append(current)
            elif current is not None and line[:1] in (" ", "+", "-"):
                content = line[1:]
                if line.startswith((" ", "-")):
                    current["old"].append(content)
                if line.startswith((" ", "+")):
                    current["new"].append(content)
        if not hunks:
            raise ValueError("Patch must contain at least one unified diff hunk")
        for hunk in reversed(hunks):
            start = hunk["start"]
            old = hunk["old"]
            if lines[start:start + len(old)] != old:
                raise ValueError("Patch context does not match the current file")
            lines[start:start + len(old)] = hunk["new"]
        target.write_text("".join(lines), encoding="utf-8", newline="")
        return {"path": path, "patched": True, "hunks": len(hunks)}

    def delete(path: str):
        target = runtime.resolve(path)
        if not target.is_file():
            raise ValueError(f"Not an existing file: {path}")
        target.unlink()
        return {"path": path, "deleted": True}

    def rename(path: str, new_path: str):
        source = runtime.resolve(path)
        destination = runtime.resolve(new_path)
        if not source.is_file():
            raise ValueError(f"Not an existing file: {path}")
        if destination.exists():
            raise ValueError(f"Destination already exists: {new_path}")
        if not destination.parent.is_dir():
            raise ValueError(f"Destination folder does not exist: {destination.parent}")
        source.rename(destination)
        return {"path": path, "new_path": new_path, "renamed": True}

    def delete_files(path: str, recursive: bool = False):
        target = runtime.resolve(path)
        if not target.is_dir():
            raise ValueError(f"Not an existing folder: {path}")
        # Preserve folders; only remove regular files under the requested folder.
        files = target.rglob("*") if recursive else target.iterdir()
        deleted = []
        for item in files:
            if item.is_file():
                item.unlink()
                deleted.append(item.relative_to(runtime.workspace).as_posix())
        return {"path": path, "deleted_count": len(deleted), "deleted": deleted, "recursive": recursive}

    def list_directory(path: str = "."):
        target = runtime.resolve(path)
        if not target.is_dir():
            raise ValueError(f"Not an existing folder: {path}")
        entries = []
        for item in sorted(target.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.casefold())):
            entry = {"name": item.name, "type": "directory" if item.is_dir() else "file"}
            if item.is_file():
                entry["size_bytes"] = item.stat().st_size
            entries.append(entry)
        return {"path": path, "entries": entries}

    for tool in [
        Tool("fs_mkdir", "Create a folder (and any missing parent folders) in the workspace.", MkdirArgs, mkdir),
        Tool("fs_write", "Create or replace a UTF-8 text file in the workspace. Parent folders are created automatically. Always pass the COMPLETE file content.", WriteArgs, write),
        Tool("fs_edit", "Update an existing workspace file by replacing one exact, unique text occurrence.", EditArgs, edit),
        Tool("fs_read", "Read a UTF-8 text file from the workspace.", ReadArgs, read),
        Tool("fs_search", "Search file names and UTF-8 text contents under a workspace folder; returns up to 200 matches.", SearchArgs, search),
        Tool("fs_patch", "Apply a unified diff containing one or more hunks to an existing workspace file. The diff must match current file context.", PatchArgs, apply_patch),
        Tool("fs_delete", "Delete one existing file in the workspace. This does not delete folders.", DeleteArgs, delete),
        Tool("fs_rename", "Rename or move one existing file within the workspace. The destination must not already exist and its parent folder must exist.", RenameArgs, rename),
        Tool("fs_delete_files", "Delete all files directly inside a workspace folder. Set recursive=true only when the user asks to include files in subfolders. Folders are preserved.", DeleteFilesArgs, delete_files),
        Tool("fs_list", "Read-only: list the files and folders directly inside a workspace folder. Use path='.' for the workspace root. This tool never changes files.", ListArgs, list_directory),
    ]:
        registry.register(tool)
