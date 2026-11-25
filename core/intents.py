"""Prompt intent patterns used to gate and route agent tool requests."""

import re

# Only action verbs indicate a workspace change request. Words like "file" or
# "folder" often appear in questions and must not enable tools by themselves.
ACTION_INTENT = re.compile(
    r"^\s*(?:(?:please|now|hey|can\s+(?:you|u)|could you|would you|i want you to)\s+){0,3}"
    r"(?:make|build|crea(?:t?e|ted|ting)|crete|generate|scaffold|write|add|edit|update|change|modify|fix|save|delete|remove|set\s+up|setup)\b",
    re.IGNORECASE,
)
# Explicit change verb + workspace object anywhere in diagnostic requests.
EXPLICIT_CHANGE_INTENT = re.compile(
    r"\b(?:write|add|edit|update|change|modify|fix|save|delete|remove|create|crete|build|scaffold|generate)\s+"
    r"(?:(?:the|a|an|new|these|those)\s+)?"
    r"(?:codes?|files?|scripts?|apps?|applications?|systems?|projects?|workspaces?|folders?|directories|"
    r"websites?|pages?|components?|features?|models?|routes?|apis?)\b",
    re.IGNORECASE,
)
INSPECT_INTENT = re.compile(
    r"\b(list|show|see|view|display|what(?:'s| is)?|which|find)\b.{0,120}\b(files?|folders?|directories|contents?)\b"
    r"|\b(files?|folders?|directories)\b.{0,120}\b(list|show|see|view|display|there|inside|in this)\b"
    r"|\b(path|location)\b.{0,100}\b(file|folder|directory|current|this)\b"
    r"|\b(file|folder|directory|current)\b.{0,100}\b(path|location)\b",
    re.IGNORECASE,
)
RUN_INTENT = re.compile(
    r"^\s*(?:(?:please|now|can\s+(?:you|u)|could you|would you)\s+){0,3}"
    r"(?:run|execute|launch|start)\b|\b(run|execute)\s+(?:it|this|that|the\s+\S+)",
    re.IGNORECASE,
)
VISIBLE_TERMINAL_INTENT = re.compile(
    r"\b(?:open|launch|use)\b.{0,60}\b(?:another|separate|new|visible)?\s*terminal\b"
    r"|\b(?:in|from)\s+(?:the\s+)?(?:other|another|separate|new)\s+terminal\b"
    r"|\bterminal\s+window\b",
    re.IGNORECASE,
)
RUN_FILE_INTENT = re.compile(
    r"\b(?:run|execute|launch|start)\b.{0,80}\b[\w./\\-]+\.(?:py|js|ts|sh|bat|cmd|ps1|exe)\b",
    re.IGNORECASE,
)
PATH_INTENT = re.compile(r"\b(path|location)\b", re.IGNORECASE)
PROCESS_LIST_INTENT = re.compile(
    r"\b(?:list|show|see|check|what(?:'s|\s+is)?|which|find)\b.{0,120}"
    r"\b(?:running|active|started)\b.{0,50}\b(?:process(?:es)?|apps?|servers?|commands?)\b"
    r"|\bwhat(?:'s|\s+is)?\s+running\b.{0,80}\b(?:folder|directory|workspace|here)\b"
    r"|\b(?:process(?:es)?|apps?|servers?)\b.{0,80}\b(?:running|active)\b",
    re.IGNORECASE,
)
PROCESS_STOP_INTENT = re.compile(
    r"^\s*(?:(?:please|now|can\s+(?:you|u)|could you|would you)\s+){0,3}"
    r"(?:stop|kill|terminate)\s+(?:(?:the|this|that|running)\s+){0,2}"
    r"(?:process|app|server|script|pid)\b",
    re.IGNORECASE,
)
CURL_REQUEST_INTENT = re.compile(
    r"\b(?:use|run|execute|send|fetch|get)\b.{0,100}\b(?:curl|endpoint|api\s+request)\b",
    re.IGNORECASE,
)
RUN_CONFIRMATION = re.compile(
    r"^\s*(?:yes|yeah|yep|sure|okay|ok|please|go ahead|proceed)(?:\s*[,!.]?\s*(?:run it|do it|proceed|go ahead|execute it))?\s*[!.]?\s*$",
    re.IGNORECASE,
)
