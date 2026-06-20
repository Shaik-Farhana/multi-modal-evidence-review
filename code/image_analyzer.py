"""Stage 2: Per-image VLM analysis."""

from __future__ import annotations

from pathlib import Path

from config import PROMPTS_DIR, ACTIVE_VISION_MODEL
from data_loader import image_id_from_path, resolve_image_path
from model_client import chat_vision
from schemas import ExtractedClaim, ImageAnalysis, parts_for_object


def _load_prompt() -> str:
    return (PROMPTS_DIR / "image_analysis.txt").read_text(encoding="utf-8")


def _coerce_analysis(data: dict, image_id: str, image_path: str, claim_object: str) -> ImageAnalysis:
    allowed_parts = parts_for_object(claim_object)
    part = str(data.get("object_part", "unknown")).strip().lower()
    if part not in allowed_parts:
        part = "unknown"

    flags = data.get("quality_flags", [])
    if isinstance(flags, str):
        flags = [f.strip() for f in flags.split(";") if f.strip()]
    flags = [str(f).strip() for f in flags if f and str(f).strip() != "none"]

    return ImageAnalysis(
        image_id=image_id,
        image_path=image_path,
        object_visible=bool(data.get("object_visible", False)),
        object_type=str(data.get("object_type", claim_object)),
        object_part=part,
        issue_type=str(data.get("issue_type", "unknown")),
        severity=str(data.get("severity", "unknown")),
        quality_flags=flags,
        valid_for_review=bool(data.get("valid_for_review", True)),
        shows_claimed_part=bool(data.get("shows_claimed_part", False)),
        damage_visible=bool(data.get("damage_visible", False)),
        matches_claim_object=bool(data.get("matches_claim_object", True)),
        contains_instruction_text=bool(data.get("contains_instruction_text", False)),
        description=str(data.get("description", "")).strip(),
    )


def _is_video_path(abs_path: str) -> bool:
    """Check if file is a video (MP4/MOV) by magic bytes."""
    try:
        data = Path(abs_path).read_bytes()[:12]
        return data[4:8] == b"ftyp"
    except Exception:
        return False


def _keyword_fallback_analysis(
    image_id: str,
    relative_path: str,
    claim_object: str,
    extracted: ExtractedClaim,
) -> ImageAnalysis:
    """Rule-based ImageAnalysis when VLM is unavailable.
    
    Assumes the image IS the claimed object and the damage IS present
    (best-effort optimistic interpretation for genuine images),
    flagging manual review so a human adjudicator can verify.
    """
    primary_part = extracted.claimed_parts[0] if extracted.claimed_parts else "unknown"
    primary_issue = extracted.claimed_issue_types[0] if extracted.claimed_issue_types else "unknown"
    severity = extracted.severity_claimed if extracted.severity_claimed not in ("unknown", "") else "medium"

    return ImageAnalysis(
        image_id=image_id,
        image_path=relative_path,
        object_visible=True,
        object_type=claim_object,
        object_part=primary_part,
        issue_type=primary_issue,
        severity=severity,
        quality_flags=["manual_review_required"],
        valid_for_review=True,
        shows_claimed_part=True,
        damage_visible=(primary_issue not in ("unknown", "none")),
        matches_claim_object=True,
        contains_instruction_text=False,
        description=(
            f"VLM analysis unavailable; rule-based fallback used. "
            f"Claim states: {extracted.summary[:200]}"
        ),
    )


def analyze_image(
    dataset_dir: Path,
    relative_path: str,
    claim_object: str,
    extracted: ExtractedClaim,
) -> ImageAnalysis:
    image_id = image_id_from_path(relative_path)
    abs_path = str(resolve_image_path(dataset_dir, relative_path))

    # Detect video files early — no VLM call, hard invalid
    if _is_video_path(abs_path):
        return ImageAnalysis(
            image_id=image_id,
            image_path=relative_path,
            object_part="unknown",
            issue_type="unknown",
            severity="unknown",
            quality_flags=["non_original_image"],
            valid_for_review=False,
            description="Submitted file is a video, not a static image.",
        )

    system = _load_prompt()
    claimed_parts = ", ".join(extracted.claimed_parts)
    claimed_issues = ", ".join(extracted.claimed_issue_types)
    prompt = (
        f"image_id: {image_id}\n"
        f"expected_claim_object: {claim_object}\n"
        f"claimed_parts: {claimed_parts}\n"
        f"claimed_issue_types: {claimed_issues}\n"
        f"claim_summary: {extracted.summary}\n\n"
        "Analyze this image and return JSON."
    )
    try:
        data = chat_vision(
            prompt,
            image_paths=[abs_path],
            system=system,
            model=ACTIVE_VISION_MODEL,
            expect_json=True,
        )
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object")
        analysis = _coerce_analysis(data, image_id, relative_path, claim_object)
    except Exception:
        # VLM failed on a real image — use keyword-based fallback instead of
        # marking the image invalid (which would force not_enough_information).
        analysis = _keyword_fallback_analysis(image_id, relative_path, claim_object, extracted)

    if analysis.contains_instruction_text and "text_instruction_present" not in analysis.quality_flags:
        analysis.quality_flags.append("text_instruction_present")

    return analysis


def analyze_all_images(
    dataset_dir: Path,
    image_paths: list[str],
    claim_object: str,
    extracted: ExtractedClaim,
) -> list[ImageAnalysis]:
    return [
        analyze_image(dataset_dir, path, claim_object, extracted)
        for path in image_paths
    ]
