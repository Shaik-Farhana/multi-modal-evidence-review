# Architecture — Multi-Modal Evidence Review

## Overview

This system verifies damage claims for **cars**, **laptops**, and **packages** by combining conversation parsing, per-image vision analysis, deterministic rules, and structured decision synthesis.

**Core principle:** Images are the primary source of truth. User history adds risk context only and never overrides clear visual evidence.

## Pipeline

```
claims.csv row
    │
    ├─► Stage 1: ClaimExtractor  (text LLM — llama3.2:3b or gemini-2.0-flash)
    │       └─► claimed_parts, claimed_issue_types, is_multi_part, summary, severity_claimed
    │
    ├─► Stage 2: ImageAnalyzer   (VLM — one call per usable image)
    │       └─► object_visible, object_part, issue_type, damage_visible, quality_flags, description
    │
    ├─► Stage 3a: EvidenceMatcher  (deterministic rules)
    │       └─► evidence_standard_met + evidence_standard_met_reason
    │
    ├─► Stage 3b: RiskAssessor    (deterministic rules + user_history.csv)
    │       └─► risk_flags: quality, cross-image, history, prompt-injection
    │
    └─► Stage 4: DecisionSynthesizer  (rules + text LLM for justification)
            └─► claim_status, issue_type, object_part, severity, supporting_image_ids, justification
```

## Backend Support

The system supports two interchangeable backends configured via `.env`:

| Backend | When Used | Models |
|---|---|---|
| **Google Gemini API** | `GEMINI_API_KEY` set | `gemini-2.0-flash` (text + vision) |
| **Local Ollama** | No API key | `llama3.2:3b` (text) + `moondream` or `qwen2.5vl:7b` (vision) |

The `model_client.py` router dispatches transparently — all pipeline modules import from it and are backend-agnostic.

**Switching backends** requires only changing `.env` — no code changes.

## Why Multi-Step?

Local VLMs are weaker than cloud APIs on complex multi-image claims. Decomposing into separate focused tasks:

1. **Text-only claim extraction** — handles multilingual chat (Hindi, Spanish, Chinese)
2. **Per-image vision analysis** — one focused VLM call per image, avoids context overload
3. **Deterministic rules** — evidence requirements, cross-image consistency, injection detection
4. **Structured synthesis** — rules-based decision + text LLM justification

Benefits:
- Auditable intermediate outputs (great for judge interview)
- Each model gets a focused, answerable task
- Rules layer is always fast and reliable regardless of model quality
- Failed VLM calls gracefully degrade (fallback extractor + `not_enough_information`)

## Model Choices

| Role | Default Model | Rationale |
|---|---|---|
| Vision (Gemini) | `gemini-2.0-flash` | Best-in-class multimodal, free tier, ~1s/image |
| Vision (Ollama) | `moondream:latest` | 1.7GB, ~30-60s/image on CPU; `qwen2.5vl:7b` if GPU available |
| Text (Gemini) | `gemini-2.0-flash` | Fast, accurate, free tier |
| Text (Ollama) | `llama3.2:3b` | 2GB, fast text extraction on CPU |
| Baseline | same VLM (single-shot) | compared in `evaluation/` for strategy selection |

## Image Format Handling

The dataset contains `.jpg`-named files that are actually:
- **JPEG** (`FF D8 FF`) — direct VLM analysis
- **PNG** (`89 50 4E 47`) — direct VLM analysis  
- **WebP** (`RIFF...WEBP`) — direct VLM analysis
- **MP4/MOV** (`00 00 ... ftyp`) — **skipped**, flagged as `non_original_image`

Detection is by magic bytes (not file extension). Video files receive a structured `valid_for_review=False` response without model calls, saving tokens.

## Decision Logic Priority

1. No usable images → `not_enough_information`, `valid_image=false`
2. Cross-image object mismatch → `not_enough_information` + `wrong_object`/`claim_mismatch`
3. Claimed part not visible → `not_enough_information` + `damage_not_visible`/`wrong_angle`
4. Visible damage contradicts claim type → `contradicted` + `claim_mismatch`
5. Visible damage matches claim → `supported`
6. Part visible but no damage → `contradicted`, `issue_type=none`

## Edge Cases Handled

| Case | Detection | Response |
|---|---|---|
| Prompt injection in chat | Keyword scan in `risk_assessor.py` | `text_instruction_present` + `manual_review_required`; claim NOT auto-approved |
| Instruction text in image | VLM `contains_instruction_text=true` | Same flags; image prompt explicitly instructs to ignore |
| Multilingual claims | Qwen2.5-VL/Gemini native + keyword fallback | Correct part/issue extraction in Hindi/Spanish/Chinese |
| Multi-part claims | `is_multi_part=true` in extraction | Relaxed cross-image part consistency check |
| User history risk | `rejected_claim >= 2` or history_flags | `user_history_risk` flag only; never overrides clear visual |
| Video submitted as image | Magic-byte header detection | `non_original_image` + `valid_for_review=False`; no model call wasted |
| Blurry/low-quality images | VLM quality_flags | Propagated to risk_flags; reduces evidence_standard_met |
| Wrong object in image | VLM `matches_claim_object=false` | `wrong_object` flag; affects claim_status |

## Evaluation Strategy

Two strategies compared on `dataset/sample_claims.csv` (21 labeled rows):

### Strategy A — Single-Shot VLM
- **One VLM call** per claim with all images + full context
- Pro: fewer API calls, simpler
- Con: weaker on multi-image claims; local models less reliable with all outputs in one shot

### Strategy B — Multi-Step Pipeline (production)
- **Full 4-stage pipeline** with separate focused model calls
- Pro: better accuracy, auditable, graceful degradation
- Con: more calls, higher latency per claim

### Metrics (composite weighted score)
| Metric | Weight |
|---|---|
| `claim_status` accuracy | 35% |
| `evidence_standard_met` accuracy | 20% |
| `issue_type` accuracy | 15% |
| `object_part` accuracy | 15% |
| `severity` accuracy | 5% |
| `risk_flags` Jaccard similarity | 10% |

## Operational Analysis

| Metric | Multi-Step (44 claims) | Single-Shot (44 claims) |
|---|---|---|
| Text calls | ~88 | 0 |
| Vision calls | ~100-130 | ~44 |
| Images processed | ~80 | ~80 |
| Cost (Gemini free tier) | **$0** | **$0** |
| Cost (local Ollama) | **$0** | **$0** |
| Latency (Gemini flash) | ~3-5s/claim | ~2-3s/claim |
| Latency (CPU Ollama) | ~30-60s/image | ~60-120s/claim |
| Rate limits | 15 RPM free tier (batched) | same |

**Retry strategy:** up to 2 retries per call with exponential backoff (1.5×, 3×).
**Batching:** sequential per-claim; images analyzed individually in multi-step mode.
**Caching opportunity:** image content hash → VLM result cache (not implemented; would save ~15% calls for repeat images).

## File Map

| Module | Responsibility |
|---|---|
| `main.py` | CLI entry point; reads claims.csv → output.csv |
| `pipeline.py` | Multi-step pipeline orchestration |
| `claim_extractor.py` | Stage 1: conversation → structured claim |
| `image_analyzer.py` | Stage 2: per-image VLM analysis |
| `evidence_matcher.py` | Stage 3a: evidence requirements rules |
| `risk_assessor.py` | Stage 3b: risk flags and history |
| `decision_synthesizer.py` | Stage 4: final status + justification |
| `single_shot.py` | Baseline single-shot strategy (evaluation) |
| `model_client.py` | Backend router (Gemini ↔ Ollama) |
| `gemini_client.py` | Gemini API client |
| `ollama_client.py` | Ollama client with image format handling |
| `schemas.py` | Pydantic models, enums, validators |
| `config.py` | Runtime config, backend selection, model aliases |
| `data_loader.py` | CSV and image path utilities |
| `csv_writer.py` | Output CSV writer with column enforcement |
| `prompts/` | System prompts for each LLM/VLM stage |
| `evaluation/main.py` | Strategy comparison + evaluation_report.md |
| `evaluation/metrics.py` | Per-field accuracy and Jaccard metrics |

## Interview Talking Points

1. **Problem framing:** Insurance-style damage verification where images are the primary source of truth — not user statements.
2. **Architecture:** 4-stage multi-step pipeline for auditability and graceful degradation.
3. **Innovation:** Per-image analysis + cross-image consistency rules + prompt-injection resistance + video file detection.
4. **Backend flexibility:** Same pipeline code works with Gemini API (cloud, fast) or Ollama (local, free).
5. **Evaluation:** Strategy A vs B on labeled sample_claims.csv with 6-metric composite score.
6. **Tradeoffs:** Local models = zero cost and privacy; cloud VLMs = much better accuracy on fine-grained damage. Would use cloud fallback for low-confidence rows in production.
7. **Future work:** Image embedding cache, confidence-based manual review queue, WebP→JPEG normalization, cloud fallback for `severity=unknown` cases.
