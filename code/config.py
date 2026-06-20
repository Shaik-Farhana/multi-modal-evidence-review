"""Runtime configuration for the evidence review agent.

Supports two backends:
  - Gemini API (recommended): set GEMINI_API_KEY in .env or environment
  - Ollama (local fallback): set OLLAMA_HOST and model names

Backend selection is automatic: Gemini is used if GEMINI_API_KEY is set.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CODE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = CODE_DIR / "prompts"

# ── Backend selection ──────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_GEMINI = bool(GEMINI_API_KEY)

if USE_GEMINI:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

# ── Gemini model names ─────────────────────────────────────────────────────────
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash")

# ── Ollama model names (used when USE_GEMINI=False) ────────────────────────────
TEXT_MODEL = os.getenv("TEXT_MODEL", "llama3.2:3b")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:7b")
SINGLE_SHOT_VISION_MODEL = os.getenv("SINGLE_SHOT_VISION_MODEL", VISION_MODEL)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# ── Active model names (resolve based on selected backend) ─────────────────────
# Use these in application modules so they work with both Gemini and Ollama.
ACTIVE_TEXT_MODEL = GEMINI_TEXT_MODEL if USE_GEMINI else TEXT_MODEL
ACTIVE_VISION_MODEL = GEMINI_VISION_MODEL if USE_GEMINI else VISION_MODEL
ACTIVE_SINGLE_SHOT_MODEL = GEMINI_VISION_MODEL if USE_GEMINI else SINGLE_SHOT_VISION_MODEL


OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
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

# ── Cost tracking ──────────────────────────────────────────────────────────────
if USE_GEMINI:
    PRICING_ASSUMPTIONS = {
        "text_model": GEMINI_TEXT_MODEL,
        "vision_model": GEMINI_VISION_MODEL,
        "cost_per_1m_input_tokens_usd": 0.0,   # free tier (1500 req/day)
        "cost_per_1m_output_tokens_usd": 0.0,
        "notes": "Google Gemini API free tier (gemini-2.0-flash); up to 1500 req/day at no cost.",
    }
else:
    PRICING_ASSUMPTIONS = {
        "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "cost_per_1m_input_tokens_usd": 0.0,
        "cost_per_1m_output_tokens_usd": 0.0,
        "notes": "Local Ollama inference; no API billing.",
    }
