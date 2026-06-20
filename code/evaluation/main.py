"""Evaluation entry point: compare single-shot vs multi-step on sample_claims.csv."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow imports from code/ when run as python evaluation/main.py
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import pandas as pd

from config import PRICING_ASSUMPTIONS, ACTIVE_TEXT_MODEL, ACTIVE_VISION_MODEL
from data_loader import load_claims
from metrics import StrategyMetrics, compare_row, format_metrics_report
from pipeline import process_claims_multi_step
from single_shot import run_single_shot_pipeline


OUTPUT_FIELDS = [
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def _evaluate_strategy(
    strategy_name: str,
    predictions: list[dict[str, str]],
    expected_rows: list[dict[str, str]],
) -> StrategyMetrics:
    metrics = StrategyMetrics(strategy_name=strategy_name)
    for pred, exp in zip(predictions, expected_rows):
        compare_row(pred, exp, metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate evidence review strategies")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("../../dataset"),
        help="Path to dataset folder",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only first N sample rows",
    )
    parser.add_argument(
        "--skip-single-shot",
        action="store_true",
        help="Skip single-shot strategy (faster dev runs)",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    sample_path = dataset_dir / "sample_claims.csv"
    if not sample_path.exists():
        print(f"Error: {sample_path} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(sample_path, dtype=str).fillna("")
    claims = load_claims(sample_path)
    if args.limit:
        claims = claims[: args.limit]
        df = df.head(args.limit)

    expected_rows = [dict(row) for _, row in df.iterrows()]
    results: list[tuple[str, StrategyMetrics, object]] = []

    print("Running multi-step strategy...")
    t0 = time.time()
    multi_outputs, multi_usage = process_claims_multi_step(claims, dataset_dir)
    multi_elapsed = time.time() - t0
    multi_preds = [o.to_csv_row() for o in multi_outputs]
    multi_metrics = _evaluate_strategy("B — Multi-step pipeline", multi_preds, expected_rows)
    results.append(("multi_step", multi_metrics, multi_usage, multi_elapsed))

    if not args.skip_single_shot:
        print("Running single-shot strategy...")
        t0 = time.time()
        single_outputs, single_usage = run_single_shot_pipeline(claims, dataset_dir)
        single_elapsed = time.time() - t0
        single_preds = [o.to_csv_row() for o in single_outputs]
        single_metrics = _evaluate_strategy("A — Single-shot VLM", single_preds, expected_rows)
        results.append(("single_shot", single_metrics, single_usage, single_elapsed))

    report_path = Path(__file__).parent / "evaluation_report.md"
    lines = [
        "# Evaluation Report\n",
        f"Dataset: `{sample_path}`\n",
        f"Rows evaluated: {len(claims)}\n",
        f"Text model: `{ACTIVE_TEXT_MODEL}` | Vision model: `{ACTIVE_VISION_MODEL}`\n",
        "## Strategy Comparison\n",
    ]
    for _, metrics, _, _ in results:
        lines.append(format_metrics_report(metrics))

    winner = max(results, key=lambda r: r[1].composite_score)
    lines.append(f"\n**Selected for production (`output.csv`): {winner[1].strategy_name}**\n")

    lines.append("\n## Operational Analysis\n")
    for name, metrics, usage, elapsed in results:
        lines.append(f"### {metrics.strategy_name}\n")
        lines.append(f"- Wall-clock runtime: {elapsed:.1f}s ({elapsed / max(len(claims), 1):.1f}s per claim)\n")
        lines.append(f"- Text model calls: {usage.text_calls}\n")
        lines.append(f"- Vision model calls: {usage.vision_calls}\n")
        lines.append(f"- Images processed: {usage.images_processed}\n")
        lines.append(f"- Est. input tokens: ~{usage.estimated_input_tokens:,}\n")
        lines.append(f"- Est. output tokens: ~{usage.estimated_output_tokens:,}\n")

    lines.append("\n### Cost & Rate Limits\n")
    lines.append(f"- Pricing: {PRICING_ASSUMPTIONS['notes']}\n")
    lines.append("- Approximate cost for full test set (44 claims, ~2 images each): **$0.00** (local Ollama)\n")
    lines.append("- Retry strategy: up to 2 retries per model call with exponential backoff\n")
    lines.append("- Batching: sequential per-claim processing; images analyzed one at a time in multi-step mode\n")
    lines.append("- TPM/RPM: limited by local GPU; no external rate limits\n")

    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")

    for _, metrics, _, _ in results:
        print(f"{metrics.strategy_name}: composite={metrics.composite_score:.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
