"""CSV and image path loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from schemas import ClaimInput


def parse_image_paths(image_paths: str) -> list[str]:
    return [p.strip() for p in image_paths.split(";") if p.strip()]


def image_id_from_path(path: str) -> str:
    return Path(path).stem


def resolve_image_path(dataset_dir: Path, relative_path: str) -> Path:
    candidate = dataset_dir / relative_path
    if candidate.exists():
        return candidate
    alt = dataset_dir.parent / relative_path
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Image not found: {relative_path} (dataset_dir={dataset_dir})")


def load_claims(csv_path: Path) -> list[ClaimInput]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    claims: list[ClaimInput] = []
    for _, row in df.iterrows():
        if not row.get("user_id"):
            continue
        claims.append(
            ClaimInput(
                user_id=str(row["user_id"]),
                image_paths=str(row["image_paths"]),
                user_claim=str(row["user_claim"]),
                claim_object=str(row["claim_object"]),
            )
        )
    return claims


def load_user_history(dataset_dir: Path) -> dict[str, dict[str, str]]:
    path = dataset_dir / "user_history.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    return {str(row["user_id"]): dict(row) for _, row in df.iterrows()}


def load_evidence_requirements(dataset_dir: Path) -> list[dict[str, str]]:
    path = dataset_dir / "evidence_requirements.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    return [dict(row) for _, row in df.iterrows()]
