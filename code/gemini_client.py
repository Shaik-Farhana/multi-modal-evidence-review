"""Gemini API client wrapper — vision + text with JSON extraction and usage tracking.

Supports gemini-2.0-flash (default) which handles multimodal natively.
Set GEMINI_API_KEY in .env or environment to enable.
Falls back to Ollama if GEMINI_API_KEY is not set.
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai

from schemas import UsageStats

_model_cache: dict[str, genai.GenerativeModel] = {}
_usage = UsageStats()


def _get_model(model_name: str) -> genai.GenerativeModel:
    if model_name not in _model_cache:
        _model_cache[model_name] = genai.GenerativeModel(model_name)
    return _model_cache[model_name]


def reset_usage() -> UsageStats:
    global _usage
    _usage = UsageStats()
    return _usage


def get_usage() -> UsageStats:
    return _usage

# Magic bytes identifying non-image (video) files
_MP4_FTYP = b"ftyp"  # bytes 4-8 of MP4


def _is_video_file(path: str) -> bool:
    """Return True if the file is an MP4/MOV masquerading as an image."""
    try:
        data = Path(path).read_bytes()[:12]
        return data[4:8] == _MP4_FTYP
    except Exception:
        return False



def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _load_image_b64(image_path: str) -> dict[str, Any] | None:
    """Load image as base64-encoded inline_data for Gemini. Returns None for video files."""
    if _is_video_file(image_path):
        return None
    path = Path(image_path)
    ext = path.suffix.lower().lstrip(".")
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "gif": "image/gif", "webp": "image/webp"}
    mime = mime_map.get(ext, "image/jpeg")
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {"inline_data": {"mime_type": mime, "data": data}}


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def chat_text(
    prompt: str,
    system: str = "",
    model: str = "gemini-2.0-flash",
    expect_json: bool = True,
    max_retries: int = 3,
) -> dict[str, Any] | str:
    """Text-only call to Gemini."""
    gem_model = _get_model(model)
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    if expect_json:
        full_prompt += "\n\nReturn ONLY valid JSON, no markdown code fences."

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = gem_model.generate_content(full_prompt)
            content = response.text or ""
            _usage.text_calls += 1
            _usage.estimated_input_tokens += _estimate_tokens(full_prompt)
            _usage.estimated_output_tokens += _estimate_tokens(content)
            if expect_json:
                return _extract_json(content)
            return content
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Gemini text call failed after retries: {last_error}")


def chat_vision(
    prompt: str,
    image_paths: list[str],
    system: str = "",
    model: str = "gemini-2.0-flash",
    expect_json: bool = True,
    max_retries: int = 3,
) -> dict[str, Any] | str:
    """Vision call to Gemini — images loaded from disk as base64."""
    gem_model = _get_model(model)
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    if expect_json:
        full_prompt += "\n\nReturn ONLY valid JSON, no markdown code fences."

    parts: list[Any] = [full_prompt]
    skipped = 0
    for img_path in image_paths:
        try:
            img_data = _load_image_b64(img_path)
            if img_data is None:
                skipped += 1  # video file — skip
            else:
                parts.append(img_data)
        except Exception:
            skipped += 1  # unreadable — skip

    # If all images were skipped (videos), return structured fallback
    if len(parts) == 1:  # only the prompt remains
        _usage.vision_calls += 1
        _usage.images_processed += len(image_paths)
        if expect_json:
            return {
                "object_visible": False,
                "object_type": "unknown",
                "object_part": "unknown",
                "issue_type": "unknown",
                "severity": "unknown",
                "quality_flags": ["non_original_image"],
                "valid_for_review": False,
                "shows_claimed_part": False,
                "damage_visible": False,
                "matches_claim_object": False,
                "contains_instruction_text": False,
                "description": "Submitted file is a video, not a static image; cannot perform automated visual review.",
            }
        return "Video file submitted; cannot perform automated visual review."

    if skipped:
        parts[0] += f"\n\n[Note: {skipped} video file(s) excluded; only static images shown.]"

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = gem_model.generate_content(parts)
            content = response.text or ""
            _usage.vision_calls += 1
            _usage.images_processed += len(image_paths)
            _usage.estimated_input_tokens += _estimate_tokens(full_prompt) + 500 * (len(parts) - 1)
            _usage.estimated_output_tokens += _estimate_tokens(content)
            if expect_json:
                return _extract_json(content)
            return content
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Gemini vision call failed after retries: {last_error}")

