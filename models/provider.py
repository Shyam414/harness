from __future__ import annotations

from typing import Any

from models.base import ChatProvider, ProviderError
from models.groq import GroqProvider
from models.huggingface import HuggingFaceProvider
from models.ollama import OllamaProvider


def create_provider(
    provider_name: str,
    model: str,
    settings: dict[str, Any],
    host: str | None = None,
) -> ChatProvider:
    """Create the one provider selected by the application's .env settings."""
    name = provider_name.strip().lower()
    if name == "ollama":
        return OllamaProvider(model, host or str(settings.get("OLLAMA_HOST") or "http://127.0.0.1:11434"))
    if name == "huggingface":
        return HuggingFaceProvider(
            model,
            token=str(settings.get("HF_TOKEN") or "") or None,
            host=str(settings.get("HF_API_BASE") or "https://router.huggingface.co/v1"),
        )
    if name == "groq":
        return GroqProvider(
            model,
            api_key=str(settings.get("GROQ_API_KEY") or "") or None,
            host=str(settings.get("GROQ_API_BASE") or "https://api.groq.com/openai/v1"),
        )
    raise ProviderError("MODEL_PROVIDER must be one of: ollama, huggingface, groq.")
