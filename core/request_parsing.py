"""Small parsers for explicit tool syntax and command snippets in prompts."""

from __future__ import annotations

import ast
import json
import re
import shlex


def redact_sensitive_text(text: str) -> str:
    """Mask credential values before persisting prompts or synthetic tool calls."""
    text = re.sub(
        r'(?i)((?:"|\b)(?:password|token|secret|api[_-]?key)(?:"|\b)\s*[:=]\s*)(?:"(?:\\.|[^"\\])*"|\S+)',
        r"\1[redacted]",
        text,
    )
    return re.sub(r"(?i)(--?(?:password|token|secret|api[-_]?key)\s+)\S+", r"\1[redacted]", text)


def curl_login_command(prompt: str) -> str | None:
    """Build a Bash-safe curl login command from an explicit URL/email/password request."""
    url_match = re.search(r"https?://[^\s\"'<>]+", prompt, re.IGNORECASE)
    email_match = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", prompt, re.IGNORECASE)
    password_match = re.search(
        r"\bpassword\s*[:=]\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s,;]+))",
        prompt,
        re.IGNORECASE,
    )
    if not (url_match and email_match and password_match):
        return None
    url = url_match.group(0).rstrip(".,;:!?)]")
    password = next((value for value in password_match.groups() if value is not None), "")
    payload = json.dumps({"email": email_match.group(0), "password": password}, separators=(",", ":"))
    return (
        f"curl -i -X POST {shlex.quote(url)} "
        f"-H {shlex.quote('Content-Type: application/json')} -d {shlex.quote(payload)}"
    )


def _invocation_end(text: str, opening: int) -> int | None:
    """Find a function call's closing parenthesis while respecting quoted strings."""
    depth = 0
    quote = None
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def explicit_read_invocation(prompt: str) -> tuple[str, dict] | None:
    """Parse explicit read-only tool syntax such as git_status(path='.')."""
    names = ("git_status", "git_diff", "fs_list", "process_list")
    for name in names:
        match = re.search(rf"\b{name}\s*\(", prompt)
        if not match:
            continue
        opening = prompt.find("(", match.start())
        end = _invocation_end(prompt, opening)
        if end is None:
            continue
        try:
            call = ast.parse(prompt[match.start():end + 1], mode="eval").body
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != name:
                continue
            if call.args:
                continue
            arguments = {item.arg: ast.literal_eval(item.value) for item in call.keywords if item.arg}
            if len(arguments) != len(call.keywords):
                continue
            return name, arguments
        except (SyntaxError, ValueError, TypeError):
            continue
    return None


def proposed_terminal_command(text: str, tool_name: str = "terminal_run") -> str | None:
    """Read a terminal tool snippet from the model's response."""
    match = re.search(rf"\b{re.escape(tool_name)}\s*\(", text)
    if not match:
        return None
    opening = text.find("(", match.start())
    end = _invocation_end(text, opening)
    if end is None:
        return None
    body = text[opening + 1:end].strip()
    if not body:
        return None
    try:
        parsed = ast.parse(f"_call({body})", mode="eval").body
        if isinstance(parsed, ast.Call):
            for item in parsed.keywords:
                if item.arg == "command":
                    value = ast.literal_eval(item.value)
                    return value if isinstance(value, str) else None
    except (SyntaxError, ValueError, TypeError):
        pass
    # Some local models print pseudo-call syntax with raw shell text. Execution
    # remains gated by the user's run/API request.
    return body
