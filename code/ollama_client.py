"""Ollama client wrapper with JSON extraction, image format handling, and usage tracking."""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import ollama

from config import MAX_RETRIES, OLLAMA_HOST, VISION_MODEL
from schemas import UsageStats

_client: ollama.Client | None = None
_usage = UsageStats()

# Magic bytes identifying non-image (video) files
_VIDEO_MAGIC = [
    bytes([0x00, 0x00, 0x00, 0x18]),  # MP4
    bytes([0x00, 0x00, 0x00, 0x1c]),  # MP4 variant
    bytes([0x00, 0x00, 0x00, 0x20]),  # MOV
    bytes([0x00, 0x00, 0x00, 0x14]),  # MP4 variant
]
_MP4_FTYP = b"ftyp"  # bytes 4-8 of MP4


def get_client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=OLLAMA_HOST)
    return _client


def reset_usage() -> UsageStats:
    global _usage
    _usage = UsageStats()
    return _usage


def get_usage() -> UsageStats:
    return _usage


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _is_video_file(path: str) -> bool:
    """Return True if the file is a video (MP4/MOV) masquerading as an image."""
    try:
        data = Path(path).read_bytes()[:12]
        if data[4:8] == _MP4_FTYP:
            return True
        for magic in _VIDEO_MAGIC:
            if data[:4] == magic:
                return True
    except Exception:
        pass
    return False


def _load_image_b64(path: str) -> str | None:
    """Load image as base64 string. Returns None if file is not a valid image."""
    if _is_video_file(path):
        return None  # skip video files
    try:
        return base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    except Exception:
        return None


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
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
    model: str | None = None,
    expect_json: bool = True,
) -> dict[str, Any] | str:
    from config import ACTIVE_TEXT_MODEL
    model = model or ACTIVE_TEXT_MODEL
    client = get_client()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat(model=model, messages=messages, format="json" if expect_json else None)
            content = response.message.content or ""
            _usage.text_calls += 1
            _usage.estimated_input_tokens += _estimate_tokens(prompt + system)
            _usage.estimated_output_tokens += _estimate_tokens(content)
            if expect_json:
                return extract_json(content)
            return content
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                if expect_json:
                    messages[-1]["content"] = prompt + "\n\nReturn ONLY valid JSON, no markdown."
    raise RuntimeError(f"Text model call failed after retries: {last_error}")


def chat_vision(
    prompt: str,
    image_paths: list[str],
    system: str = "",
    model: str | None = None,
    expect_json: bool = True,
) -> dict[str, Any] | str:
    from config import ACTIVE_VISION_MODEL
    model = model or ACTIVE_VISION_MODEL
    client = get_client()

    # Load images as base64, skipping video files
    valid_images: list[str] = []
    skipped_videos: list[str] = []
    for p in image_paths:
        b64 = _load_image_b64(p)
        if b64 is None:
            skipped_videos.append(p)
        else:
            valid_images.append(b64)

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})

    # If no valid images, return early with a structured response indicating video/invalid
    if not valid_images:
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

    msg_content = prompt
    if skipped_videos:
        msg_content += f"\n\n[Note: {len(skipped_videos)} video file(s) were excluded; only {len(valid_images)} static image(s) are shown.]"
    messages.append({"role": "user", "content": msg_content, "images": valid_images})

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat(model=model, messages=messages, format="json" if expect_json else None)
            content = response.message.content or ""
            _usage.vision_calls += 1
            _usage.images_processed += len(image_paths)
            _usage.estimated_input_tokens += _estimate_tokens(prompt + system) + 500 * len(valid_images)
            _usage.estimated_output_tokens += _estimate_tokens(content)
            if expect_json:
                return extract_json(content)
            return content
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(2.0 * (attempt + 1))
                if expect_json:
                    messages[-1]["content"] = prompt + "\n\nReturn ONLY valid JSON, no markdown."
    raise RuntimeError(f"Vision model call failed after retries: {last_error}")


def list_available_models() -> list[str]:
    try:
        return [m.model for m in get_client().list().models]
    except Exception:
        return []
