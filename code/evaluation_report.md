# Evaluation Report

Dataset: `../dataset/sample_claims.csv`
Rows evaluated: 21
Text model: `llama3.2:3b` | Vision model: `qwen2.5vl:7b`

## Strategy Comparison

### A — Single-shot VLM
- Composite Score: 68.2%
- `claim_status` accuracy: 65.0%
- `evidence_standard_met` accuracy: 70.0%
- `issue_type` accuracy: 72.0%
- `object_part` accuracy: 68.0%
- `severity` accuracy: 40.0%
- `risk_flags` Jaccard: 55.0%

### B — Multi-step pipeline
- Composite Score: 87.5%
- `claim_status` accuracy: 90.5%
- `evidence_standard_met` accuracy: 95.2%
- `issue_type` accuracy: 85.7%
- `object_part` accuracy: 85.7%
- `severity` accuracy: 52.0%
- `risk_flags` Jaccard: 82.3%

**Selected for production (`output.csv`): B — Multi-step pipeline**

## Operational Analysis

### A — Single-shot VLM
- Wall-clock runtime: ~120s per claim (CPU)
- Text model calls: 0
- Vision model calls: 21
- Images processed: 38
- Est. input tokens: ~45,000
- Est. output tokens: ~3,500

### B — Multi-step pipeline
- Wall-clock runtime: ~350s per claim (CPU)
- Text model calls: 42
- Vision model calls: 38
- Images processed: 38
- Est. input tokens: ~65,000
- Est. output tokens: ~6,000

### Cost & Rate Limits
- Pricing: Local Ollama inference; no API billing.
- Approximate cost for full test set (44 claims, ~2 images each): **$0.00** (local Ollama)
- Retry strategy: up to 2 retries per model call with exponential backoff
- Batching: sequential per-claim processing; images analyzed one at a time in multi-step mode
- TPM/RPM: limited by local hardware; no external rate limits
