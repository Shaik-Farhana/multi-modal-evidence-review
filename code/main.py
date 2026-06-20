"""Main entry point for the evidence review agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import run_multi_step_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review Agent")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("../dataset"),
        help="Path to dataset folder containing CSVs and images/",
    )
    parser.add_argument(
        "--claims",
        type=str,
        default="claims.csv",
        help="Claims CSV filename inside dataset-dir",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../output.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N claims (for dev/testing)",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        print(f"Error: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    claims_path = dataset_dir / args.claims
    if not claims_path.exists():
        print(f"Error: claims file not found: {claims_path}", file=sys.stderr)
        return 1

    print(f"Processing {claims_path} -> {args.output}")
    usage = run_multi_step_pipeline(
        dataset_dir=dataset_dir,
        claims_csv=args.claims,
        output_path=args.output.resolve(),
        limit=args.limit,
    )
    print(
        f"Done. Vision calls={usage.vision_calls}, text calls={usage.text_calls}, "
        f"images={usage.images_processed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
