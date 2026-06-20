"""Stage 1: Extract structured claim from conversation."""

from __future__ import annotations

from pathlib import Path

from config import PROMPTS_DIR, ACTIVE_TEXT_MODEL
from model_client import chat_text
from schemas import ExtractedClaim, parts_for_object


def _load_prompt() -> str:
    return (PROMPTS_DIR / "claim_extraction.txt").read_text(encoding="utf-8")


def _normalize_part(part: str | dict, claim_object: str) -> str:
    if isinstance(part, dict):
        part = part.get("part") or part.get("name") or part.get("value") or str(part)
    allowed = parts_for_object(claim_object)
    part = str(part).strip().lower().replace(" ", "_")
    if part in allowed:
        return part
    aliases = {
        "front_bumper": "front_bumper", "rear_bumper": "rear_bumper",
        "back_bumper": "rear_bumper", "bumper": "front_bumper",
        "windshield": "windshield", "front_glass": "windshield",
        "screen": "screen", "display": "screen", "pantalla": "screen",
        "keyboard": "keyboard", "trackpad": "trackpad",
        "hinge": "hinge", "mirror": "side_mirror", "side_mirror": "side_mirror",
        "headlight": "headlight", "taillight": "taillight", "back_light": "taillight",
        "package_corner": "package_corner", "corner": "package_corner",
        "seal": "seal", "label": "label", "contents": "contents",
        "box": "box", "hood": "hood", "door": "door", "lid": "lid",
        "body": "body", "item": "item", "package_side": "package_side",
    }
    if part in aliases and aliases[part] in allowed:
        return aliases[part]
    return "unknown"


def _normalize_issue(issue: str | dict) -> str:
    if isinstance(issue, dict):
        issue = issue.get("type") or issue.get("name") or issue.get("value") or str(issue)
    issue = str(issue).strip().lower().replace(" ", "_")
    allowed = {
        "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
        "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
    }
    aliases = {
        "shatter": "glass_shatter", "shattered": "glass_shatter",
        "broken": "broken_part", "missing": "missing_part",
        "torn": "torn_packaging", "crushed": "crushed_packaging",
        "water": "water_damage", "liquid": "water_damage", "oil": "stain",
    }
    if issue in allowed:
        return issue
    if issue in aliases:
        return aliases[issue]
    return "unknown"


def extract_claim(user_claim: str, claim_object: str) -> ExtractedClaim:
    system = _load_prompt()
    prompt = (
        f"claim_object: {claim_object}\n\n"
        f"Conversation:\n{user_claim}\n\n"
        "Extract the damage claim as JSON."
    )
    try:
        data = chat_text(prompt, system=system, model=ACTIVE_TEXT_MODEL, expect_json=True)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object")
    except Exception:
        return _fallback_extract(user_claim, claim_object)

    parts = [_normalize_part(p, claim_object) for p in data.get("claimed_parts", []) if p]
    issues = [_normalize_issue(i) for i in data.get("claimed_issue_types", []) if i]
    parts = [p for p in parts if p != "unknown"] or ["unknown"]
    issues = issues or ["unknown"]

    return ExtractedClaim(
        claimed_parts=parts,
        claimed_issue_types=issues,
        is_multi_part=bool(data.get("is_multi_part")) or len(parts) > 1,
        summary=str(data.get("summary", "")).strip() or _fallback_summary(user_claim),
        severity_claimed=str(data.get("severity_claimed", "unknown")),
    )


def _fallback_extract(user_claim: str, claim_object: str) -> ExtractedClaim:
    text = user_claim.lower()
    parts: list[str] = []
    issues: list[str] = []

    part_keywords = {
        "car": {
            "front_bumper": ["front bumper", "front side", "bumper ke upar"],
            "rear_bumper": ["rear bumper", "back bumper", "back of the car"],
            "door": ["door", "door panel"],
            "hood": ["hood", "hail"],
            "windshield": ["windshield", "front glass", "front glass"],
            "side_mirror": ["side mirror", "mirror"],
            "headlight": ["headlight", "left headlight"],
            "taillight": ["taillight", "back light", "rear light"],
            "fender": ["fender"], "body": ["body panel", "car body"],
        },
        "laptop": {
            "screen": ["screen", "display", "pantalla"],
            "keyboard": ["keyboard", "keys", "keycap"],
            "trackpad": ["trackpad"],
            "hinge": ["hinge"],
            "corner": ["corner"],
            "lid": ["lid"],
            "body": ["body", "outer body"],
        },
        "package": {
            "package_corner": ["corner", "package corner"],
            "seal": ["seal", "torn", "opened", "open"],
            "label": ["label"],
            "contents": ["contents", "missing", "inside", "item inside"],
            "box": ["box", "package", "cardboard"],
            "package_side": ["side", "surface", "wet"],
        },
    }

    for part, keywords in part_keywords.get(claim_object, {}).items():
        if any(kw in text for kw in keywords):
            parts.append(part)

    issue_keywords = {
        "dent": ["dent", "dab gaya"],
        "scratch": ["scratch", "scrape", "mark"],
        "crack": ["crack", "cracked"],
        "glass_shatter": ["shatter", "shattered"],
        "broken_part": ["broken", "broke", "toot gaya"],
        "missing_part": ["missing", "keycaps came off", "keys missing"],
        "torn_packaging": ["torn", "open", "opened", "seal"],
        "crushed_packaging": ["crushed", "crush"],
        "water_damage": ["water", "wet"],
        "stain": ["stain", "oil"],
    }
    for issue, keywords in issue_keywords.items():
        if any(kw in text for kw in keywords):
            issues.append(issue)

    if not parts:
        parts = ["unknown"]
    if not issues:
        issues = ["unknown"]

    return ExtractedClaim(
        claimed_parts=parts,
        claimed_issue_types=issues,
        is_multi_part=len(parts) > 1,
        summary=_fallback_summary(user_claim),
        severity_claimed="unknown",
    )


def _fallback_summary(user_claim: str) -> str:
    return user_claim.split("|")[-1].strip()[:200]
