"""Unified model client — routes to Gemini API or Ollama based on config.

Usage:
    from model_client import chat_text, chat_vision, get_usage, reset_usage

Backend is selected automatically:
  - Gemini API if GEMINI_API_KEY is set in .env or environment
  - Ollama otherwise (requires local Ollama server)
"""

from __future__ import annotations

from typing import Any

from config import USE_GEMINI
from schemas import UsageStats

if USE_GEMINI:
    import gemini_client as _backend
else:
    import ollama_client as _backend  # type: ignore[no-redef]


def chat_text(
    prompt: str,
    system: str = "",
    model: str | None = None,
    expect_json: bool = True,
) -> dict[str, Any] | str:
    if model is not None:
        return _backend.chat_text(prompt, system=system, model=model, expect_json=expect_json)
    return _backend.chat_text(prompt, system=system, expect_json=expect_json)


def chat_vision(
    prompt: str,
    image_paths: list[str],
    system: str = "",
    model: str | None = None,
    expect_json: bool = True,
) -> dict[str, Any] | str:
    if model is not None:
        return _backend.chat_vision(prompt, image_paths=image_paths, system=system, model=model, expect_json=expect_json)
    return _backend.chat_vision(prompt, image_paths=image_paths, system=system, expect_json=expect_json)


def get_usage() -> UsageStats:
    return _backend.get_usage()


def reset_usage() -> UsageStats:
    return _backend.reset_usage()
