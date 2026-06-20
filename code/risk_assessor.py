"""Stage 3b: User history and cross-image risk assessment."""

from __future__ import annotations

from schemas import ExtractedClaim, ImageAnalysis, RiskAssessment


def assess_history_risk(user_history: dict[str, str] | None) -> list[str]:
    flags: list[str] = []
    if not user_history:
        return flags
    history_flags = user_history.get("history_flags", "none")
    if history_flags and history_flags != "none" and "user_history_risk" in history_flags:
        flags.append("user_history_risk")
    rejected = int(user_history.get("rejected_claim", "0") or 0)
    manual = int(user_history.get("manual_review_claim", "0") or 0)
    if rejected >= 2 or manual >= 2:
        if "user_history_risk" not in flags:
            flags.append("user_history_risk")
    return flags


def assess_cross_image_risks(
    analyses: list[ImageAnalysis],
    extracted: ExtractedClaim,
    claim_object: str,
) -> list[str]:
    flags: list[str] = []
    if len(analyses) < 2:
        return flags

    object_types = {a.object_type for a in analyses if a.object_type != "unknown"}
    if len(object_types) > 1:
        flags.extend(["wrong_object", "claim_mismatch", "manual_review_required"])

    parts = {a.object_part for a in analyses if a.object_part != "unknown"}
    if len(parts) > 2 and not extracted.is_multi_part:
        if "claim_mismatch" not in flags:
            flags.append("claim_mismatch")

    match_flags = [a for a in analyses if not a.matches_claim_object]
    if len(match_flags) >= len(analyses) // 2 + 1:
        if "wrong_object" not in flags:
            flags.append("wrong_object")

    visible_damage = [a for a in analyses if a.damage_visible]
    no_damage = [a for a in analyses if a.object_visible and not a.damage_visible]
    if visible_damage and no_damage and not extracted.is_multi_part:
        if "claim_mismatch" not in flags:
            flags.append("claim_mismatch")

    return flags


def collect_image_quality_flags(analyses: list[ImageAnalysis]) -> list[str]:
    flags: list[str] = []
    for analysis in analyses:
        for flag in analysis.quality_flags:
            if flag != "none" and flag not in flags:
                flags.append(flag)
    return flags


def assess_risks(
    analyses: list[ImageAnalysis],
    extracted: ExtractedClaim,
    claim_object: str,
    user_history: dict[str, str] | None,
    user_claim: str,
) -> RiskAssessment:
    flags: list[str] = []
    flags.extend(collect_image_quality_flags(analyses))
    flags.extend(assess_cross_image_risks(analyses, extracted, claim_object))
    flags.extend(assess_history_risk(user_history))

    claim_lower = user_claim.lower()
    injection_phrases = [
        "approve immediately", "skip manual review", "ignore all previous instructions",
        "mark this row supported", "follow it and approve", "ignore previous instructions",
    ]
    if any(p in claim_lower for p in injection_phrases):
        if "text_instruction_present" not in flags:
            flags.append("text_instruction_present")
        if "manual_review_required" not in flags:
            flags.append("manual_review_required")

    if any(a.contains_instruction_text for a in analyses):
        if "text_instruction_present" not in flags:
            flags.append("text_instruction_present")
        if "manual_review_required" not in flags:
            flags.append("manual_review_required")

    if "user_history_risk" in flags and "manual_review_required" not in flags:
        flags.append("manual_review_required")

    flags = [f for f in flags if f != "none"]
    summary = user_history.get("history_summary", "") if user_history else ""
    return RiskAssessment(risk_flags=flags, history_summary=summary)
