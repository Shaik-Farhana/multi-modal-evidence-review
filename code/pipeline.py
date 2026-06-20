"""Multi-step and single-shot claim processing pipelines."""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from claim_extractor import extract_claim
from csv_writer import write_output_csv
from data_loader import (
    load_claims,
    load_evidence_requirements,
    load_user_history,
    parse_image_paths,
)
from decision_synthesizer import synthesize_decision
from evidence_matcher import assess_evidence
from image_analyzer import analyze_all_images
from model_client import get_usage, reset_usage
from risk_assessor import assess_risks
from schemas import ClaimInput, ClaimOutput, UsageStats


def process_claim_multi_step(
    claim: ClaimInput,
    dataset_dir: Path,
    user_history_map: dict[str, dict[str, str]],
    evidence_requirements: list[dict[str, str]],
) -> ClaimOutput:
    extracted = extract_claim(claim.user_claim, claim.claim_object)
    image_paths = parse_image_paths(claim.image_paths)
    analyses = analyze_all_images(dataset_dir, image_paths, claim.claim_object, extracted)
    evidence = assess_evidence(claim.claim_object, extracted, analyses, evidence_requirements)
    history = user_history_map.get(claim.user_id)
    risks = assess_risks(analyses, extracted, claim.claim_object, history, claim.user_claim)
    return synthesize_decision(claim, extracted, analyses, evidence, risks)


def process_claims_multi_step(
    claims: list[ClaimInput],
    dataset_dir: Path,
) -> tuple[list[ClaimOutput], UsageStats]:
    reset_usage()
    user_history_map = load_user_history(dataset_dir)
    evidence_requirements = load_evidence_requirements(dataset_dir)
    outputs: list[ClaimOutput] = []
    for claim in tqdm(claims, desc="Processing claims"):
        outputs.append(
            process_claim_multi_step(claim, dataset_dir, user_history_map, evidence_requirements)
        )
    return outputs, get_usage()


def run_multi_step_pipeline(
    dataset_dir: Path,
    claims_csv: str,
    output_path: Path,
    limit: int | None = None,
) -> UsageStats:
    claims = load_claims(dataset_dir / claims_csv)
    if limit:
        claims = claims[:limit]
    outputs, usage = process_claims_multi_step(claims, dataset_dir)
    write_output_csv(outputs, output_path)
    return usage
