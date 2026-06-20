"""Fast deterministic pipeline — runs entirely without LLM/VLM.

Produces output.csv in under 30 seconds on any hardware by using:
  - Keyword extraction for Stage 1 (claim_extractor._fallback_extract)
  - Keyword-based image analysis for Stage 2 (no VLM calls)
  - Full deterministic rules for Stage 3a, 3b, 4
  - Fallback text justification (no LLM calls)

Run:
    python fast_pipeline.py --dataset-dir ../dataset --output ../output.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from claim_extractor import _fallback_extract
from csv_writer import write_output_csv
from data_loader import (
    load_claims,
    load_evidence_requirements,
    load_user_history,
    parse_image_paths,
    resolve_image_path,
)
from decision_synthesizer import _fallback_justification, _format_risk_flags, _normalize_output_enums, _determine_status
from evidence_matcher import assess_evidence
from image_analyzer import _is_video_path, _keyword_fallback_analysis
from risk_assessor import assess_risks
from schemas import ClaimOutput


def _is_video(dataset_dir: Path, relative_path: str) -> bool:
    try:
        abs_path = str(resolve_image_path(dataset_dir, relative_path))
        return _is_video_path(abs_path)
    except Exception:
        return False


def run_fast_pipeline(dataset_dir: Path, claims_file: str, output_path: Path) -> None:
    claims_path = dataset_dir / claims_file
    claims = load_claims(claims_path)
    requirements = load_evidence_requirements(dataset_dir)
    history = load_user_history(dataset_dir)

    outputs: list[ClaimOutput] = []

    for claim in claims:
        # Stage 1: keyword extraction (no LLM)
        extracted = _fallback_extract(claim.user_claim, claim.claim_object)

        # Stage 2: image analysis (no VLM — keyword fallback per image)
        image_paths = parse_image_paths(claim.image_paths)
        analyses = []
        for rel_path in image_paths:
            from data_loader import image_id_from_path
            image_id = image_id_from_path(rel_path)
            if _is_video(dataset_dir, rel_path):
                from schemas import ImageAnalysis
                analyses.append(ImageAnalysis(
                    image_id=image_id,
                    image_path=rel_path,
                    object_part="unknown",
                    issue_type="unknown",
                    severity="unknown",
                    quality_flags=["non_original_image"],
                    valid_for_review=False,
                    description="Submitted file is a video, not a static image.",
                ))
            else:
                analyses.append(_keyword_fallback_analysis(image_id, rel_path, claim.claim_object, extracted))

        # Stage 3a: evidence matching
        evidence = assess_evidence(claim.claim_object, extracted, analyses, requirements)

        # Stage 3b: risk assessment
        user_hist = history.get(claim.user_id, {})
        risks = assess_risks(analyses, extracted, claim.claim_object, user_hist, claim.user_claim)

        # Stage 4: decision synthesis (no LLM)
        claim_status, issue_type, object_part, severity, supporting_ids = _determine_status(
            extracted, analyses, evidence, risks
        )
        issue_type, object_part, severity = _normalize_output_enums(
            claim.claim_object, issue_type, object_part, severity
        )

        valid_image = any(a.valid_for_review for a in analyses)
        justification = _fallback_justification(claim_status, analyses, supporting_ids, object_part, issue_type)

        outputs.append(ClaimOutput(
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
        ))

    write_output_csv(outputs, output_path)
    print(f"Done. {len(outputs)} claims written to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast deterministic pipeline (no LLM/VLM)")
    parser.add_argument("--dataset-dir", default="../dataset", type=Path)
    parser.add_argument("--claims", default="claims.csv")
    parser.add_argument("--output", default="../output.csv", type=Path)
    args = parser.parse_args()
    run_fast_pipeline(args.dataset_dir, args.claims, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
