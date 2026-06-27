# Redrob Hackathon — Intelligent Candidate Ranking System

This repository contains a deterministic, multi-stage heuristic ranking engine designed to evaluate candidates for the Senior AI Engineer role at Redrob.

## Architecture Overview

This system processes 100,000 resumes in under 2 minutes using CPU-only resources. It avoids generic keyword-matching traps by prioritizing substance over keywords:

- **Word-Boundary Safety:** Custom regex helpers prevent substring false positives (e.g. `"ann"` matching inside `"planning"`, `"bert"` inside `"robert"`). All IR/ML keyword checks use word-boundary-aware matching.
- **Substance-over-Keywords Scoring:** Career history is weighted far above skills-section listings. Bare keyword presence is insufficient — candidates need evidence of deployment, evaluation, and ownership to score well.
- **Disclaimed Ownership Detection:** Identifies and discounts candidates who explicitly disclaim production ownership (e.g. "deployment was handled by another team", "I wouldn't call myself an ML specialist").
- **Multi-Stage Pipeline:** Hard Gates → Core Relevance Index (CRI) → Raw Fit Score → Calibration → Confidence Multiplier → Final Ranking.

The CRI uses a max-of-two-paths formula to fairly score both ranking/retrieval specialists and pure recommendation-systems engineers, since the JD explicitly lists both as valid profiles.

## Requirements

- **Python:** 3.9+
- **Dependencies:** None. This system uses only Python standard libraries to guarantee zero-dependency deployment and maximum efficiency.

## Setup & Reproduction

1. Clone this repository.
2. Ensure your `candidates.jsonl` file is located in the repository root.
3. Run the following command to generate the ranked output:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

This produces `submission.csv` containing the top 100 ranked candidates with scores and reasoning strings. Runtime is approximately 2 minutes on a standard CPU with 16 GB RAM.

## Output Format

The output CSV contains four columns: `candidate_id`, `rank`, `score`, `reasoning`. Candidates ranked below the gate threshold are explicitly labeled as mandatory filler in their reasoning string, per the submission spec's sample format.
