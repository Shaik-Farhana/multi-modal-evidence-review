# Multi-Modal Evidence Review Agent

A production-grade, multi-step damage-claim verification system built for the **HackerRank Orchestrate June 2026** hackathon.

This system acts as an automated insurance claims adjuster. Given a damage claim conversation, submitted images, user history, and strict evidence requirements, the system analyzes the visual evidence to decide whether it **supports**, **contradicts**, or provides **not enough information** to verify the claim.

---

## 🌟 Key Features

- **Multi-Modal Pipeline**: Combines text Large Language Models (LLMs) for complex conversation parsing with Vision-Language Models (VLMs) for strict visual verification.
- **Dual-Backend Support**: Seamlessly switch between the lightning-fast **Google Gemini API** (cloud) and **Ollama** (local, privacy-first) without changing any code.
- **Smart Image Handling**: Automatically detects fake/disguised image formats (like MP4 videos named `.jpg`) via magic bytes, saving compute tokens by skipping invalid files. WebP and PNG formats are fully supported.
- **Prompt Injection Defense**: Deterministic rules scan for and flag adversarial text in conversations (e.g., "approve this claim immediately") and instructions hidden within images.
- **Evidence-First Decision Making**: The system explicitly trusts *visual evidence* over user history. User risk flags trigger manual reviews but do not falsify valid photographic evidence.

---

## 🚀 Quick Start

### 1. Prerequisites

You need Python 3.10+ and either a free Google Gemini API key OR local Ollama.

```powershell
# Clone or enter the code directory
cd code

# Install dependencies
pip install -r requirements.txt
```

### 2. Choose Your Backend

**Option A: Google Gemini API (Recommended - Fast & Free)**
Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
Open `.env` and set:
```env
GEMINI_API_KEY=your_api_key_here
```

**Option B: Local Ollama (Privacy First)**
Install [Ollama](https://ollama.com) and pull the models:
```powershell
ollama pull qwen2.5vl:7b
ollama pull llama3.2:3b
```
*(Make sure `GEMINI_API_KEY` is commented out in your `.env` file)*

### 3. Generate Predictions

```powershell
# Full test run on all claims (produces output.csv)
python main.py --dataset-dir ../dataset --claims claims.csv --output ../output.csv

# Dev/debug run (process only the first 3 claims)
python main.py --dataset-dir ../dataset --claims claims.csv --output ../output_dev.csv --limit 3
```

### 4. Evaluate (Strategy Comparison)

Run the automated evaluation on the labeled sample dataset to compare the Multi-Step pipeline against a Single-Shot baseline.

```powershell
python evaluation/main.py --dataset-dir ../dataset
```
*The detailed metrics report will be written to `evaluation_report.md`.*

---

## 🏗 System Architecture

The pipeline processes each claim through 4 distinct stages to ensure auditability, deterministic policy enforcement, and graceful degradation.

```text
Input: claims.csv row
    │
    ├─► Stage 1: ClaimExtractor  (Text LLM)
    │       Reads the chat and extracts structured data: parts, issue types, severity.
    │       Handles multilingual conversations gracefully.
    │
    ├─► Stage 2: ImageAnalyzer   (Vision VLM)
    │       Analyzes EACH image individually to prevent context overload.
    │       Outputs: damage visibility, quality flags, part identification.
    │
    ├─► Stage 3a: EvidenceMatcher  (Deterministic Rules)
    │       Cross-references VLM findings against evidence_requirements.csv.
    │       Outputs: evidence_standard_met (True/False) + reason.
    │
    ├─► Stage 3b: RiskAssessor    (Deterministic Rules)
    │       Checks user_history.csv, detects cross-image mismatches, and
    │       flags prompt injections.
    │
    └─► Stage 4: DecisionSynthesizer  (Rules + Text LLM)
            Synthesizes the final decision (supported/contradicted/not_enough_info).
            Writes a cohesive justification grounded purely in the images.
```

### Why Multi-Step?
Instead of passing everything into one giant prompt (Single-Shot), decomposing the problem means:
1. **Higher Accuracy**: Local VLMs perform poorly when juggling 15 JSON fields across multiple images.
2. **Auditability**: We can inspect exactly what the VLM saw in Image 2 vs Image 3.
3. **Graceful Fallback**: If a VLM call fails, the deterministic rules still output a safe `not_enough_information` state rather than crashing.

---

## ⚖️ System Design: Pros & Cons

### Pros
- **Highly Modular:** The unified `model_client.py` makes swapping out models or entire backend providers (Gemini vs. Ollama) trivial.
- **Deterministic Safeguards:** The `RiskAssessor` and `EvidenceMatcher` ensure that AI hallucinations don't bypass hard company policies.
- **Cost Efficient:** Can run at exactly $0.00 using local Ollama or the Gemini Free Tier.
- **Resilient File Handling:** Protects against bad data (e.g., MP4 videos masquerading as images) before it wastes VLM compute.

### Cons
- **Latency (Local):** Running `qwen2.5vl:7b` locally on a CPU can take 3-5 minutes per image. (Mitigated by the Gemini cloud integration which takes ~3 seconds).
- **Complexity:** Managing state across 4 stages requires robust schemas (`schemas.py`) and error handling compared to a single prompt script.
- **Missing Embeddings Cache:** Currently, if a user uploads the exact same image in two different claims, it is processed twice. An image hashing layer could improve performance.

---

## 🛡 Edge Case Handling

| Scenario | System Handling |
|---|---|
| **Fake Images (MP4 videos)** | Magic bytes detect video headers (`ftyp`). Bypasses the VLM, saving compute, and flags `non_original_image`. |
| **Prompt Injection in Chat** | Keyword scans in `risk_assessor.py` flag `text_instruction_present`. The claim is routed to manual review and is never auto-approved. |
| **Multilingual Claims** | `qwen2.5vl` and Gemini natively support Hindi, Spanish, etc. A keyword fallback acts as a safety net. |
| **Blurry/Obscured Images** | VLM detects quality issues. Triggers `evidence_standard_met=False` leading to `not_enough_information`. |
| **Wrong Object Uploaded** | VLM identifies the object does not match the claim. Flags `wrong_object` and rejects the evidence. |

---

## 📁 File Structure

| File / Folder | Purpose |
|---|---|
| `main.py` | CLI entry point for processing the dataset. |
| `pipeline.py` | Orchestrates the 4 stages of the multi-step pipeline. |
| `claim_extractor.py` | Stage 1: Text extraction logic. |
| `image_analyzer.py` | Stage 2: Vision analysis logic. |
| `evidence_matcher.py` | Stage 3a: Deterministic evidence logic. |
| `risk_assessor.py` | Stage 3b: Risk and history logic. |
| `decision_synthesizer.py`| Stage 4: Final output generation. |
| `model_client.py` | Backend router (switches between Gemini and Ollama). |
| `gemini_client.py` | Google Gemini API implementation. |
| `ollama_client.py` | Local Ollama API implementation (with magic byte checks). |
| `schemas.py` | Pydantic models ensuring strict data typing. |
| `config.py` | Environment variable and constant configurations. |
| `data_loader.py` | Pandas-based CSV and file loaders. |
| `csv_writer.py` | Enforces exact column ordering for `output.csv`. |
| `prompts/` | Plaintext system prompts for the LLMs/VLMs. |
| `evaluation/` | Scripts and metrics for comparing system accuracy. |

---

## 📦 Hackathon Submission

To build the submission zip file locally:

```powershell
# Run this from the repository root
python -c "
import zipfile, os
def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        if '__pycache__' in root: continue
        for file in files:
            if file.endswith('.pyc') or file == '.env': continue
            file_path = os.path.join(root, file)
            ziph.write(file_path, os.path.relpath(file_path, path))
with zipfile.ZipFile('../code.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipdir('.', zipf)
print('code.zip created.')
"
```
*Note: Do not include `dataset/` or your `.env` file with API keys in the final submission!*
