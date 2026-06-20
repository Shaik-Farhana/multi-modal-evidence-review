# Multi-Modal Evidence Review Agent

> An automated AI system that verifies damage claims using images, conversation transcripts, user history, and evidence requirements — built for the HackerRank Orchestrate June 2026 hackathon.

Given a support chat about a damaged **car**, **laptop**, or **package**, plus submitted photos, the system decides:

- ✅ `supported` — images clearly show the claimed damage
- ❌ `contradicted` — images contradict the claim
- ⚠️ `not_enough_information` — evidence is missing, invalid, or inconclusive

---

## 🌟 Key Features

| Feature | Detail |
|---|---|
| **4-Stage Pipeline** | Text extraction → Vision analysis → Policy rules → Decision synthesis |
| **Dual Backend** | Google Gemini API (cloud, fast) *or* Ollama (local, offline, $0) |
| **Smart Fallback** | When VLM unavailable, uses keyword extraction to still produce meaningful predictions |
| **Video Detection** | Detects MP4/MOV files disguised as `.jpg` via magic bytes — skips them without wasting tokens |
| **Prompt Injection Defense** | Catches adversarial text like *"approve this claim"* in both conversations and images |
| **Multilingual** | Handles Hindi, Spanish, Chinese and more via Gemini / Qwen2.5-VL natively |
| **Zero Cost** | Runs entirely free via Ollama locally or Gemini's free tier (1,500 req/day) |

---

## 🏗 Architecture

```
claims.csv row
    │
    ├─► Stage 1 — ClaimExtractor  (Text LLM: llama3.2:3b / gemini-2.0-flash)
    │       Parses the support conversation → structured JSON
    │       Extracts: claimed_parts, issue_types, severity, is_multi_part
    │       Handles multilingual text; keyword fallback if LLM unavailable
    │
    ├─► Stage 2 — ImageAnalyzer  (Vision LLM: qwen2.5vl:7b / gemini-2.0-flash)
    │       One VLM call per image (prevents context overload)
    │       Extracts: object_visible, damage_visible, quality_flags, issue_type, severity
    │       Detects MP4 videos via magic bytes → flags non_original_image
    │       Keyword fallback if VLM unavailable → valid_for_review=True still
    │
    ├─► Stage 3a — EvidenceMatcher  (Deterministic rules)
    │       Reads evidence_requirements.csv
    │       → evidence_standard_met (True/False) + reason
    │
    ├─► Stage 3b — RiskAssessor  (Deterministic rules)
    │       Reads user_history.csv
    │       Checks: cross-image mismatches, fraud history, prompt injection
    │       → risk_flags list
    │
    └─► Stage 4 — DecisionSynthesizer  (Rules + Text LLM)
            Priority logic: evidence → risk → visual match → final status
            → claim_status, issue_type, object_part, severity, justification
```

### Why Multi-Step instead of Single-Shot?

| | Single-Shot | Multi-Step (this system) |
|---|---|---|
| Accuracy | Lower — LLM must juggle 15 fields + images at once | Higher — each model has one focused task |
| Auditability | Black box | Inspect each stage's JSON independently |
| Failure mode | Hard crash or hallucinated JSON | Graceful degradation with keyword fallback |
| Cost | Fewer calls | More calls, but each is cheaper/smaller |

---

## ⚖️ Design Tradeoffs

### ✅ Strengths
- **Backend-agnostic**: `model_client.py` transparently routes to Gemini or Ollama — swap providers by changing one `.env` line
- **Deterministic guardrails**: `EvidenceMatcher` and `RiskAssessor` enforce hard policies that LLMs cannot override
- **Resilient to bad data**: Handles mixed image formats (JPEG, PNG, WebP) and silently skips videos
- **Cost = $0**: Works with Gemini free tier or fully offline with Ollama

### ⚠️ Limitations
- **Local CPU latency**: `qwen2.5vl:7b` on CPU-only hardware takes ~5 min/image (use Gemini for speed)
- **Severity precision**: Fine-grained severity (`low` vs `medium` vs `high`) is the weakest prediction — needs a fine-tuned model
- **No image cache**: Identical images submitted across different claims are re-analysed each time

---

## 🛡 Edge Cases Handled

| Scenario | Detection | Response |
|---|---|---|
| MP4/MOV file named `.jpg` | Magic bytes `[4:8] == b'ftyp'` | Skip VLM, flag `non_original_image`, `valid_image=false` |
| Prompt injection in chat | Keyword scan in `risk_assessor.py` | Flag `text_instruction_present` + `manual_review_required`; never auto-approve |
| Instructions embedded in image | VLM `contains_instruction_text=true` | Same flags; VLM prompt explicitly instructs to ignore |
| Multilingual claim | Gemini/Qwen native + keyword fallback | Correct part/issue extraction in Hindi, Spanish, etc. |
| Blurry / obscured image | VLM quality flags | `evidence_standard_met=False` → `not_enough_information` |
| Wrong object uploaded | VLM `matches_claim_object=false` | `wrong_object` flag → reject evidence |
| Multi-part claim | `is_multi_part=true` | Relaxed cross-image part consistency check |
| High-risk user history | History flags (≥2 rejections) | `user_history_risk` added — never overrides clear visual evidence |
| VLM completely unavailable | `except Exception` in image_analyzer | `_keyword_fallback_analysis()` — produces valid prediction from claim text |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- One of: a [Gemini API key](https://aistudio.google.com/apikey) **or** [Ollama](https://ollama.com) installed

```bash
git clone https://github.com/Shaik-Farhana/multi-modal-evidence-review.git
cd multi-modal-evidence-review/code
pip install -r requirements.txt
```

### Option A — Google Gemini (Recommended: fast, free, no GPU needed)

```bash
# Create a .env file
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Option B — Local Ollama (Fully offline, privacy-first)

```bash
ollama pull qwen2.5vl:7b   # vision model
ollama pull llama3.2:3b    # text model
# Leave GEMINI_API_KEY unset in .env
```

### Option C — No models at all (Fastest, deterministic only)

```bash
# Uses keyword extraction + rule-based decisions. Runs in ~6 seconds.
python fast_pipeline.py --dataset-dir ../dataset --output ../output.csv
```

---

## ▶️ Running the Pipeline

```bash
# Full multi-step pipeline (requires Gemini key or Ollama)
python main.py --dataset-dir ../dataset --claims claims.csv --output ../output.csv

# Debug: process only first 3 claims
python main.py --dataset-dir ../dataset --claims claims.csv --output ../output.csv --limit 3

# Fast deterministic pipeline (no LLM/VLM needed — 6 seconds)
python fast_pipeline.py --dataset-dir ../dataset --output ../output.csv

# Evaluate: compare Multi-Step vs Single-Shot on labeled sample data
python evaluation/main.py --dataset-dir ../dataset
```

---

## 📊 Results (44 Test Claims)

| Metric | Value |
|---|---|
| Total claims processed | 44 |
| `supported` | 37 (84%) |
| `not_enough_information` | 7 (16%) |
| `evidence_standard_met = True` | 41 / 44 |
| `valid_image = True` | 42 / 44 |
| MP4 videos detected & skipped | 2 |

**Issue type breakdown:** crack (9), dent (8), broken_part (6), missing_part (4), scratch (4), water_damage (2), crushed_packaging (2), glass_shatter (1), stain (1), unknown (7)

---

## 📋 Output Schema

`output.csv` — exact column order:

| Column | Allowed Values |
|---|---|
| `user_id` | from input |
| `image_paths` | from input |
| `user_claim` | from input |
| `claim_object` | `car` · `laptop` · `package` |
| `evidence_standard_met` | `true` · `false` |
| `evidence_standard_met_reason` | free text |
| `risk_flags` | semicolon-separated (see below) |
| `issue_type` | `dent` · `scratch` · `crack` · `glass_shatter` · `broken_part` · `missing_part` · `torn_packaging` · `crushed_packaging` · `water_damage` · `stain` · `none` · `unknown` |
| `object_part` | object-specific (e.g. `front_bumper`, `screen`, `seal`) |
| `claim_status` | `supported` · `contradicted` · `not_enough_information` |
| `claim_status_justification` | image-grounded free text |
| `supporting_image_ids` | semicolon-separated IDs or `none` |
| `valid_image` | `true` · `false` |
| `severity` | `none` · `low` · `medium` · `high` · `unknown` |

**Risk flag values:** `none` · `blurry_image` · `cropped_or_obstructed` · `low_light_or_glare` · `wrong_angle` · `wrong_object` · `wrong_object_part` · `damage_not_visible` · `claim_mismatch` · `possible_manipulation` · `non_original_image` · `text_instruction_present` · `user_history_risk` · `manual_review_required`

---

## 📁 File Map

```
code/
├── main.py                   # CLI entry point (full pipeline)
├── fast_pipeline.py          # Deterministic-only pipeline (~6 seconds)
├── pipeline.py               # Multi-step orchestrator
│
├── claim_extractor.py        # Stage 1: conversation → structured claim
├── image_analyzer.py         # Stage 2: VLM per-image analysis + fallback
├── evidence_matcher.py       # Stage 3a: policy rules
├── risk_assessor.py          # Stage 3b: fraud & injection detection
├── decision_synthesizer.py   # Stage 4: final decision + justification
│
├── model_client.py           # Backend router (Gemini ↔ Ollama)
├── gemini_client.py          # Google Gemini API client
├── ollama_client.py          # Ollama client (magic-byte video detection)
│
├── schemas.py                # Pydantic models & enums
├── config.py                 # Config loaded from .env
├── data_loader.py            # CSV & image path utilities
├── csv_writer.py             # Enforces exact output column order
│
├── prompts/                  # System prompts for each LLM/VLM stage
│   ├── claim_extraction.txt
│   ├── image_analysis.txt
│   ├── decision_synthesis.txt
│   └── single_shot.txt
│
├── evaluation/
│   ├── main.py               # Strategy comparison script
│   └── metrics.py            # Accuracy & Jaccard scoring
│
├── evaluation_report.md      # Results: Multi-step vs Single-shot
├── ARCHITECTURE.md           # Deep-dive design document
└── requirements.txt
```

---

## 🔧 Configuration (`.env`)

```env
# ── Gemini API (recommended) ───────────────────────────
# Get a free key at https://aistudio.google.com/apikey
GEMINI_API_KEY=your_key_here

# ── Local Ollama (fallback when no API key) ────────────
TEXT_MODEL=llama3.2:3b
VISION_MODEL=qwen2.5vl:7b
SINGLE_SHOT_VISION_MODEL=qwen2.5vl:7b
```

The backend is selected **automatically** — if `GEMINI_API_KEY` is set, Gemini is used; otherwise Ollama is used. No code changes required.

---

## 📈 Evaluation

Two strategies are compared on the labeled `sample_claims.csv` (21 rows with ground truth):

| Strategy | Composite Score |
|---|---|
| A — Single-shot VLM | 68.2% |
| B — Multi-step pipeline | **87.5%** |

Metrics: `claim_status` (35%) · `evidence_standard_met` (20%) · `issue_type` (15%) · `object_part` (15%) · `severity` (5%) · `risk_flags` Jaccard (10%)

---

*Built with Python · Pydantic · Pandas · Google Gemini API · Ollama*
