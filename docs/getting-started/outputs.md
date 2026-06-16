# Reading the output

Every run writes the same provenance set regardless of configuration, plus one file
per requested export format:

```
output/
  sft_alpaca.jsonl          SFT data in Alpaca format
  sft_sharegpt.jsonl        SFT data in ShareGPT conversation format
  dpo.jsonl                 DPO preference pairs (only when preference task used)
  grpo.jsonl                GRPO rollouts (only when grpo task used)
  ppo.jsonl                 PPO prompts (only when ppo exporter included)
  corpus.jsonl              Raw source chunks (only when corpus exporter included)
  rejected.jsonl            Every rejected sample with a structured reason string
  manifest.json             Pipeline config hash, stage counts, rejection breakdown
  dataset_card.md           Human-readable run summary
  checksums.txt             SHA-256 for all output files
  diagnostic_summary.json   Failure mode counts, recovery rate (when probe active)
```

## manifest.json

Per-stage sample counts: how many entered, passed, and were rejected at each stage:

```json
{
  "pipeline_config_hash": "a3f7c2d1",
  "stage_counts": {
    "SchemaGate":          {"input_count": 1000, "output_count": 980,  "probe_recovered": 0, "rejected_count": 20},
    "QAGenerationTask":    {"input_count": 980,  "output_count": 2850, "rejected_count": 90},
    "HallucinationGate":   {"input_count": 2850, "output_count": 2210, "probe_recovered": 0, "rejected_count": 640}
  }
}
```

Gates also record `probe_recovered`, the number of rejected samples the diagnostic
probe repaired and returned to the pipeline. Exporter stages record a single `exported_count`.

## rejected.jsonl

Every line carries a structured `rejection_reason` and the step that rejected it.
In this example the hallucination gate's grounding check (its "contract") scored
the answer 0.43, below the configured threshold:

```json
{"id": "...", "instruction": "...", "rejection_reason": "hallucination_contract_failed:0.43", "rejecting_step": "HallucinationGate"}
```

Use these to tune thresholds, not to debug code.

## Inspecting results in code

```python
result.print_summary()
# ────────────────────────────────────────────
#   passed   :      2,210
#   rejected :        640
#   time     :      87.3s
#   output   : output/qa
# ────────────────────────────────────────────

result.sample(n=3)   # print first 3 passed samples

len(result.passed)   # list of DataSample
len(result.rejected) # list of RejectedSample

# per-stage counts
result.stage_counts["HallucinationGate"]
# → {"input_count": 2850, "output_count": 2210, "probe_recovered": 0, "rejected_count": 640}
```

Next: the [guides](../guides/index.md) cover each pipeline stage in depth.
