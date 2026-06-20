"""Stage 4: Merge signals into final claim output."""

from __future__ import annotations

import json

from config import PROMPTS_DIR, ACTIVE_TEXT_MODEL
from model_client import chat_text
from schemas import (
    ClaimInput,
    ClaimOutput,
    EvidenceAssessment,
    ExtractedClaim,
    ImageAnalysis,
    RiskAssessment,
    parts_for_object,
)


def _issues_match(claimed: str, visible: str) -> bool:
    if visible in ("none", "unknown"):
        return visible == "none" and claimed in ("none", "unknown")
    compatible = {
        "dent": {"dent", "scratch"},
        "scratch": {"scratch", "dent"},
        "crack": {"crack", "glass_shatter", "broken_part"},
        "glass_shatter": {"glass_shatter", "crack", "broken_part"},
        "broken_part": {"broken_part", "crack", "missing_part"},
        "missing_part": {"missing_part", "broken_part", "none"},
        "torn_packaging": {"torn_packaging"},
        "crushed_packaging": {"crushed_packaging", "dent"},
        "water_damage": {"water_damage", "stain"},
        "stain": {"stain", "water_damage"},
    }
    if claimed == visible:
        return True
    return visible in compatible.get(claimed, set())


def _parts_match(claimed_parts: list[str], visible_part: str) -> bool:
    if "unknown" in claimed_parts:
        return True
    if visible_part in claimed_parts:
        return True
    related = {
        "front_bumper": {"front_bumper", "body", "fender"},
        "rear_bumper": {"rear_bumper", "body", "quarter_panel"},
        "door": {"door", "body", "quarter_panel"},
        "screen": {"screen", "lid"},
        "package_corner": {"package_corner", "box"},
        "seal": {"seal", "box", "package_side"},
        "contents": {"contents", "item", "box"},
    }
    for claimed in claimed_parts:
        if visible_part in related.get(claimed, {claimed}):
            return True
    return False


def _pick_best_analysis(analyses: list[ImageAnalysis], extracted: ExtractedClaim) -> ImageAnalysis | None:
    scored: list[tuple[int, ImageAnalysis]] = []
    for a in analyses:
        if not a.valid_for_review:
            continue
        score = 0
        if a.damage_visible:
            score += 3
        if a.shows_claimed_part:
            score += 2
        if _parts_match(extracted.claimed_parts, a.object_part):
            score += 2
        if a.object_visible:
            score += 1
        if "blurry_image" in a.quality_flags:
            score -= 2
        scored.append((score, a))
    if not scored:
        return analyses[0] if analyses else None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _determine_status(
    extracted: ExtractedClaim,
    analyses: list[ImageAnalysis],
    evidence: EvidenceAssessment,
    risks: RiskAssessment,
) -> tuple[str, str, str, str, list[str]]:
    """Returns claim_status, issue_type, object_part, severity, supporting_ids."""
    usable = [a for a in analyses if a.valid_for_review]
    best = _pick_best_analysis(analyses, extracted)

    cross_image_flags = {"wrong_object", "claim_mismatch"}
    if cross_image_flags.intersection(risks.risk_flags) and len(analyses) > 1:
        supporting = [a.image_id for a in analyses]
        return (
            "not_enough_information",
            best.issue_type if best else "unknown",
            best.object_part if best else extracted.claimed_parts[0],
            "unknown",
            supporting,
        )

    if not evidence.evidence_standard_met or not usable:
        return (
            "not_enough_information",
            best.issue_type if best else "unknown",
            best.object_part if best else extracted.claimed_parts[0],
            "unknown",
            [],
        )

    if not best:
        return (
            "not_enough_information",
            "unknown",
            extracted.claimed_parts[0],
            "unknown",
            [],
        )

    primary_issue = extracted.claimed_issue_types[0]
    primary_part = extracted.claimed_parts[0]

    if not best.shows_claimed_part and "unknown" not in extracted.claimed_parts:
        if "damage_not_visible" not in risks.risk_flags:
            risks.risk_flags.append("damage_not_visible")
        if "wrong_angle" not in risks.risk_flags and best.object_visible:
            risks.risk_flags.append("wrong_angle")
        return (
            "not_enough_information",
            "unknown",
            primary_part,
            "unknown",
            [],
        )

    visible_issue = best.issue_type
    visible_part = best.object_part if best.object_part != "unknown" else primary_part
    severity = best.severity if best.severity != "unknown" else "medium"

    if best.damage_visible and _issues_match(primary_issue, visible_issue) and _parts_match(extracted.claimed_parts, visible_part):
        supporting = [best.image_id]
        for a in analyses:
            if a.image_id != best.image_id and a.damage_visible and a.valid_for_review:
                if _issues_match(primary_issue, a.issue_type):
                    supporting.append(a.image_id)
        return ("supported", visible_issue, visible_part, severity, supporting)

    if best.object_visible and not best.damage_visible and visible_issue == "none":
        if "claim_mismatch" not in risks.risk_flags:
            risks.risk_flags.append("claim_mismatch")
        return ("contradicted", "none", visible_part, "none", [best.image_id])

    if best.damage_visible and not _issues_match(primary_issue, visible_issue):
        if "claim_mismatch" not in risks.risk_flags:
            risks.risk_flags.append("claim_mismatch")
        sev = severity if severity != "none" else "low"
        return ("contradicted", visible_issue, visible_part, sev, [best.image_id])

    if best.damage_visible:
        return ("supported", visible_issue, visible_part, severity, [best.image_id])

    return (
        "not_enough_information",
        visible_issue,
        visible_part,
        "unknown",
        [],
    )


def _format_risk_flags(flags: list[str]) -> str:
    cleaned = [f for f in flags if f and f != "none"]
    return ";".join(dict.fromkeys(cleaned)) if cleaned else "none"


def _generate_justification(
    claim: ClaimInput,
    extracted: ExtractedClaim,
    analyses: list[ImageAnalysis],
    claim_status: str,
    issue_type: str,
    object_part: str,
    supporting_ids: list[str],
) -> str:
    system = (PROMPTS_DIR / "decision_synthesis.txt").read_text(encoding="utf-8")
    context = {
        "claim_status": claim_status,
        "issue_type": issue_type,
        "object_part": object_part,
        "supporting_image_ids": supporting_ids,
        "claim_summary": extracted.summary,
        "image_analyses": [
            {"image_id": a.image_id, "description": a.description, "issue_type": a.issue_type}
            for a in analyses
        ],
    }
    prompt = (
        f"claim_object: {claim.claim_object}\n"
        f"user_claim excerpt: {claim.user_claim[:500]}\n"
        f"decision context: {json.dumps(context)}\n"
        "Write the justification JSON."
    )
    try:
        data = chat_text(prompt, system=system, model=ACTIVE_TEXT_MODEL, expect_json=True)
        if isinstance(data, dict) and data.get("claim_status_justification"):
            return str(data["claim_status_justification"]).strip()
    except Exception:
        pass
    return _fallback_justification(claim_status, analyses, supporting_ids, object_part, issue_type)


def _fallback_justification(
    claim_status: str,
    analyses: list[ImageAnalysis],
    supporting_ids: list[str],
    object_part: str,
    issue_type: str,
) -> str:
    part_label = object_part.replace("_", " ")
    if claim_status == "supported" and supporting_ids:
        desc = next((a.description for a in analyses if a.image_id == supporting_ids[0]), "")
        ids = " and ".join(supporting_ids)
        return f"The image set supports the claim because {desc or f'{issue_type} on the {part_label} is visible in {ids}'}."
    if claim_status == "contradicted":
        desc = next((a.description for a in analyses if a.description), "")
        return f"The submitted images do not support the claim as stated; {desc or f'visible evidence shows {issue_type} on the {part_label} instead'}."
    return f"The submitted images do not provide enough evidence to verify the claimed {part_label} damage."


def _normalize_output_enums(
    claim_object: str,
    issue_type: str,
    object_part: str,
    severity: str,
) -> tuple[str, str, str]:
    allowed_issues = {
        "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
        "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
    }
    allowed_severity = {"none", "low", "medium", "high", "unknown"}
    allowed_parts = parts_for_object(claim_object)

    if issue_type not in allowed_issues:
        issue_type = "unknown"
    if object_part not in allowed_parts:
        object_part = "unknown"
    if severity not in allowed_severity:
        severity = "unknown"
    return issue_type, object_part, severity


def synthesize_decision(
    claim: ClaimInput,
    extracted: ExtractedClaim,
    analyses: list[ImageAnalysis],
    evidence: EvidenceAssessment,
    risks: RiskAssessment,
) -> ClaimOutput:
    claim_status, issue_type, object_part, severity, supporting_ids = _determine_status(
        extracted, analyses, evidence, risks
    )
    issue_type, object_part, severity = _normalize_output_enums(
        claim.claim_object, issue_type, object_part, severity
    )

    valid_image = any(a.valid_for_review for a in analyses)
    if any(f in risks.risk_flags for f in ("non_original_image", "possible_manipulation")):
        valid_image = False

    justification = _generate_justification(
        claim, extracted, analyses, claim_status, issue_type, object_part, supporting_ids
    )

    return ClaimOutput(
        user_id=claim.user_id,
        image_paths=claim.image_paths,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object,
        evidence_standard_met=evidence.evidence_standard_met,
        evidence_standard_met_reason=evidence.evidence_standard_met_reason,
        risk_flags=_format_risk_flags(risks.risk_flags),
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=justification,
        supporting_image_ids=";".join(supporting_ids) if supporting_ids else "none",
        valid_image=valid_image,
        severity=severity,
    )
