# Automated Eval Dataset Generator from Production Logs

A system that mines production LLM logs, finds the interesting / edge-case /
failure-mode interactions, and automatically converts them into labeled
evaluation test cases — building an ever-growing, production-representative eval
dataset without manual curation.

## Why this exists

The hardest part of AI evaluation isn't the harness, it's the *dataset*.
Hand-curated golden sets go stale. This solves the data-supply problem by turning
real production traffic into eval data automatically — a force multiplier.

## Architecture

```
src/ingestion/schema.py      unified log schema + regex PII redaction
                              (email/phone/SSN/card) applied before storage
src/ingestion/sampling.py    three strategies: uniform, stratified (by feature/model),
                              signal-boosted (oversamples downvotes/retries/high-latency)
src/classify/cluster.py      HDBSCAN semantic clustering + outlier identification
src/classify/difficulty.py   heuristic difficulty tagger (simple/moderate/hard/adversarial)
src/labeling/auto_label.py   dual-run LLM labeling; confidence = agreement between runs
src/labeling/dataset.py      eval-case model, embedding dedup, confidence-based routing
                              (high -> dataset, low -> human review queue)
src/runner/eval_runner.py    assertion-based scoring, regression detection, coverage report
src/pipeline.py              nightly pipeline: sample -> cluster -> label -> dedup -> add
```

## Design decisions

- **PII is redacted at ingestion, before anything is stored.** Production prompts
  contain emails, phone numbers, card numbers. Regex redaction runs on the way in so
  raw PII never lands in the eval store.
- **Signal-boosted sampling oversamples the interesting tail.** Uniform sampling
  buries the valuable cases (thumbs-down, retries, high-latency) in a sea of routine
  traffic. The signal-boosted sampler weights 70% toward those — they're the eval
  cases that actually teach you something.
- **Label confidence comes from agreement between two independent labeling runs.**
  A single LLM label isn't trustworthy enough to auto-add. Running the labeler twice
  and measuring agreement gives a cheap confidence signal: high-agreement labels
  auto-add to the dataset, low-agreement ones route to a human review queue — the
  dataset grows without sacrificing quality.
- **New candidates are deduped against the existing dataset by embedding.** Before
  adding a case, it's checked for near-duplicates (cosine > 0.92) so the dataset
  doesn't fill with slight rephrasings of the same interaction.
- **The eval harness scores by assertions, not an LLM.** Each case carries
  `must_contain` and `must_not_contain` token lists (hallucination traps), so
  re-running the suite is deterministic and free — no LLM cost per eval run.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
# Full install (needs a C toolchain for hdbscan):
pip install -r requirements.txt
# Or core-only (skips clustering; everything else works):
pip install -r requirements-core.txt
cp .env.example .env      # OPENAI_API_KEY for the labeling stage
```

## Run the pipeline

```bash
python -m src.pipeline data/sample_logs.json
# samples logs -> clusters -> auto-labels -> dedups -> writes data/eval_dataset.jsonl
# (low-confidence labels go to data/review_queue.jsonl)
```

## Tests

```bash
pytest tests/ -v
```

14 tests covering PII redaction, all three samplers, difficulty tagging,
confidence routing, embedding dedup (duplicate vs. novel), and the eval runner
(assertion scoring, regression detection, coverage) — all offline via a
deterministic fake embedder, no API key required.

## Docker

```bash
docker build -t eval-dataset-generator .
docker run --env-file .env -v $(pwd)/data:/app/data eval-dataset-generator
```

## Status

Phases 1-4 complete (ingestion+PII+sampling, clustering+difficulty, dual-run
auto-labeling+dedup+routing, eval runner+regression+coverage). The clustering
stage lazy-imports HDBSCAN so the rest runs core-only. Phase 5's Streamlit
curation dashboard and the cron scheduler are wired conceptually (pipeline.py is
the cron entry point) but the review UI itself is not built.
