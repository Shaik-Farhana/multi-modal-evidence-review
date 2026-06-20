"""Output CSV writer that enforces exact column order from config."""

from __future__ import annotations

import csv
from pathlib import Path

from config import OUTPUT_COLUMNS
from schemas import ClaimOutput


def write_output_csv(outputs: list[ClaimOutput], output_path: Path) -> None:
    """Write ClaimOutput rows to output_path with exact column order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for output in outputs:
            row = output.to_csv_row()
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})
