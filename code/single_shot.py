"""Single-shot baseline strategy for evaluation comparison."""

from __future__ import annotations

import json
from pathlib import Path

from config import PROMPTS_DIR, ACTIVE_SINGLE_SHOT_MODEL
from data_loader import load_user_history, parse_image_paths, resolve_image_path
from model_client import chat_vision, reset_usage, get_usage
from schemas import ClaimInput, ClaimOutput, UsageStats
from csv_writer import write_output_csv


def _load_prompt() -> str:
    return (PROMPTS_DIR / "single_shot.txt").read_text(encoding="utf-8")


def process_claim_single_shot(
    claim: ClaimInput,
    dataset_dir: Path,
    user_history_map: dict[str, dict[str, str]],
) -> ClaimOutput:
    system = _load_prompt()
    history = user_history_map.get(claim.user_id, {})
    image_paths = parse_image_paths(claim.image_paths)
    abs_paths = [str(resolve_image_path(dataset_dir, p)) for p in image_paths]

    prompt = (
        f"user_id: {claim.user_id}\n"
        f"claim_object: {claim.claim_object}\n"
        f"image_paths: {claim.image_paths}\n"
        f"user_history: {json.dumps(history)}\n\n"
        f"Conversation:\n{claim.user_claim}\n\n"
        "Analyze all images and return the complete output JSON."
    )

    try:
        data = chat_vision(
            prompt,
            image_paths=abs_paths,
            system=system,
            model=ACTIVE_SINGLE_SHOT_MODEL,
            expect_json=True,
        )
        if not isinstance(data, dict):
            raise ValueError("Expected JSON")
    except Exception:
        from pipeline import process_claim_multi_step
        from data_loader import load_evidence_requirements
        return process_claim_multi_step(
            claim, dataset_dir, user_history_map, load_evidence_requirements(dataset_dir)
        )

    def _bool(val: object) -> bool:
        if isinstance(val, bool):
            return val
        return str(val).lower() == "true"

    return ClaimOutput(
        user_id=claim.user_id,
        image_paths=claim.image_paths,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object,
        evidence_standard_met=_bool(data.get("evidence_standard_met", False)),
        evidence_standard_met_reason=str(data.get("evidence_standard_met_reason", "")),
        risk_flags=str(data.get("risk_flags", "none")),
        issue_type=str(data.get("issue_type", "unknown")),
        object_part=str(data.get("object_part", "unknown")),
        claim_status=str(data.get("claim_status", "not_enough_information")),
        claim_status_justification=str(data.get("claim_status_justification", "")),
        supporting_image_ids=str(data.get("supporting_image_ids", "none")),
        valid_image=_bool(data.get("valid_image", True)),
        severity=str(data.get("severity", "unknown")),
    )


def run_single_shot_pipeline(
    claims: list[ClaimInput],
    dataset_dir: Path,
) -> tuple[list[ClaimOutput], UsageStats]:
    reset_usage()
    user_history_map = load_user_history(dataset_dir)
    outputs = [
        process_claim_single_shot(claim, dataset_dir, user_history_map)
        for claim in claims
    ]
    return outputs, get_usage()
