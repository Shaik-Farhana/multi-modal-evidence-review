"""Pydantic schemas and enums matching problem_statement.md allowed values."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ClaimObject(str, Enum):
    CAR = "car"
    LAPTOP = "laptop"
    PACKAGE = "package"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"


class IssueType(str, Enum):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    BROKEN_PART = "broken_part"
    MISSING_PART = "missing_part"
    TORN_PACKAGING = "torn_packaging"
    CRUSHED_PACKAGING = "crushed_packaging"
    WATER_DAMAGE = "water_damage"
    STAIN = "stain"
    NONE = "none"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class RiskFlag(str, Enum):
    NONE = "none"
    BLURRY_IMAGE = "blurry_image"
    CROPPED_OR_OBSTRUCTED = "cropped_or_obstructed"
    LOW_LIGHT_OR_GLARE = "low_light_or_glare"
    WRONG_ANGLE = "wrong_angle"
    WRONG_OBJECT = "wrong_object"
    WRONG_OBJECT_PART = "wrong_object_part"
    DAMAGE_NOT_VISIBLE = "damage_not_visible"
    CLAIM_MISMATCH = "claim_mismatch"
    POSSIBLE_MANIPULATION = "possible_manipulation"
    NON_ORIGINAL_IMAGE = "non_original_image"
    TEXT_INSTRUCTION_PRESENT = "text_instruction_present"
    USER_HISTORY_RISK = "user_history_risk"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


CAR_PARTS = {
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
}
LAPTOP_PARTS = {
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base", "body", "unknown",
}
PACKAGE_PARTS = {
    "box", "package_corner", "package_side", "seal", "label", "contents", "item", "unknown",
}


def parts_for_object(claim_object: str) -> set[str]:
    if claim_object == ClaimObject.CAR.value:
        return CAR_PARTS
    if claim_object == ClaimObject.LAPTOP.value:
        return LAPTOP_PARTS
    return PACKAGE_PARTS


class ClaimInput(BaseModel):
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str


class ExtractedClaim(BaseModel):
    claimed_parts: list[str] = Field(default_factory=list)
    claimed_issue_types: list[str] = Field(default_factory=list)
    is_multi_part: bool = False
    summary: str = ""
    severity_claimed: str = "unknown"


class ImageAnalysis(BaseModel):
    image_id: str
    image_path: str
    object_visible: bool = False
    object_type: str = "unknown"
    object_part: str = "unknown"
    issue_type: str = "unknown"
    severity: str = "unknown"
    quality_flags: list[str] = Field(default_factory=list)
    valid_for_review: bool = True
    shows_claimed_part: bool = False
    damage_visible: bool = False
    matches_claim_object: bool = True
    contains_instruction_text: bool = False
    description: str = ""


class EvidenceAssessment(BaseModel):
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    applicable_requirement_ids: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    risk_flags: list[str] = Field(default_factory=list)
    history_summary: str = ""


class ClaimOutput(BaseModel):
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: bool
    severity: str

    @field_validator("risk_flags", "supporting_image_ids", mode="before")
    @classmethod
    def normalize_semicolon_fields(cls, value: object) -> str:
        if value is None:
            return "none"
        text = str(value).strip()
        return text if text else "none"

    def to_csv_row(self) -> dict[str, str | bool]:
        return {
            "user_id": self.user_id,
            "image_paths": self.image_paths,
            "user_claim": self.user_claim,
            "claim_object": self.claim_object,
            "evidence_standard_met": str(self.evidence_standard_met).lower(),
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": self.risk_flags,
            "issue_type": self.issue_type,
            "object_part": self.object_part,
            "claim_status": self.claim_status,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": self.supporting_image_ids,
            "valid_image": str(self.valid_image).lower(),
            "severity": self.severity,
        }


class UsageStats(BaseModel):
    text_calls: int = 0
    vision_calls: int = 0
    images_processed: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0

    def merge(self, other: UsageStats) -> None:
        self.text_calls += other.text_calls
        self.vision_calls += other.vision_calls
        self.images_processed += other.images_processed
        self.estimated_input_tokens += other.estimated_input_tokens
        self.estimated_output_tokens += other.estimated_output_tokens
