"""Stage 3a: Match claims to evidence requirements."""

from __future__ import annotations

from schemas import EvidenceAssessment, ExtractedClaim, ImageAnalysis


ISSUE_TO_APPLIES = {
    "dent": "dent or scratch",
    "scratch": "dent or scratch",
    "crack": "crack, broken, or missing part",
    "glass_shatter": "crack, broken, or missing part",
    "broken_part": "crack, broken, or missing part",
    "missing_part": "contents or inner item",
    "torn_packaging": "crushed, torn, or seal damage",
    "crushed_packaging": "crushed, torn, or seal damage",
    "water_damage": "water, stain, or label damage",
    "stain": "water, stain, or label damage",
}


def _select_requirements(
    claim_object: str,
    extracted: ExtractedClaim,
    requirements: list[dict[str, str]],
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    issue_families = {ISSUE_TO_APPLIES.get(i, "general claim review") for i in extracted.claimed_issue_types}

    for req in requirements:
        obj = req.get("claim_object", "all")
        applies = req.get("applies_to", "")
        if obj not in ("all", claim_object):
            continue
        if applies == "general claim review" or applies == "reviewability" or applies == "multi-image rows":
            selected.append(req)
            continue
        if any(family in applies or applies in family for family in issue_families):
            selected.append(req)
        elif "contents" in extracted.claimed_parts and "contents" in applies:
            selected.append(req)
        elif any(p in applies for p in extracted.claimed_parts):
            selected.append(req)

    if not selected:
        selected = [r for r in requirements if r.get("requirement_id") == "REQ_GENERAL_OBJECT_PART"]
    return selected


def _image_satisfies_requirement(
    analysis: ImageAnalysis,
    extracted: ExtractedClaim,
    claim_object: str,
) -> bool:
    if not analysis.valid_for_review:
        return False
    if not analysis.matches_claim_object and analysis.object_type not in (claim_object, "unknown"):
        return False

    claimed_parts = set(extracted.claimed_parts)
    if "unknown" not in claimed_parts:
        if analysis.shows_claimed_part:
            return True
        if analysis.object_part in claimed_parts:
            return True
        if analysis.object_visible and analysis.damage_visible:
            return True
        return False

    return analysis.object_visible and (analysis.damage_visible or analysis.object_part != "unknown")


def assess_evidence(
    claim_object: str,
    extracted: ExtractedClaim,
    analyses: list[ImageAnalysis],
    requirements: list[dict[str, str]],
) -> EvidenceAssessment:
    applicable = _select_requirements(claim_object, extracted, requirements)
    req_ids = [r["requirement_id"] for r in applicable]

    usable = [a for a in analyses if a.valid_for_review]
    if not usable:
        return EvidenceAssessment(
            evidence_standard_met=False,
            evidence_standard_met_reason="No submitted image is usable for automated review.",
            applicable_requirement_ids=req_ids,
        )

    satisfying = [a for a in usable if _image_satisfies_requirement(a, extracted, claim_object)]

    if satisfying:
        parts_str = ", ".join(extracted.claimed_parts)
        best = satisfying[0]
        reason = (
            f"The {best.object_part.replace('_', ' ')} is visible and "
            f"the claimed {parts_str} condition can be evaluated from the submitted images."
        )
        return EvidenceAssessment(
            evidence_standard_met=True,
            evidence_standard_met_reason=reason,
            applicable_requirement_ids=req_ids,
        )

    if any(not a.shows_claimed_part for a in usable):
        return EvidenceAssessment(
            evidence_standard_met=False,
            evidence_standard_met_reason=(
                "The submitted images do not clearly show the claimed part needed to verify the damage."
            ),
            applicable_requirement_ids=req_ids,
        )

    return EvidenceAssessment(
        evidence_standard_met=False,
        evidence_standard_met_reason=(
            "The image set does not provide sufficient visual evidence to evaluate the claim."
        ),
        applicable_requirement_ids=req_ids,
    )
