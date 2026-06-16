# Train with AlignTune

[AlignTune](https://github.com/Lexsi-Labs/aligntune) is Lexsi Labs' fine-tuning
library. CuratorKIT's Alpaca and DPO exports match the dataset shapes AlignTune's
SFT and RL trainers consume, so the two compose into a curate-then-train workflow.

## 1. Curate

Generate and gate a dataset, with splits:

```python
from curatorkit import Curator, CuratorConfig

result = Curator(CuratorConfig(
    dataset                 = "handbook.pdf",
    llm_model               = "openai/gpt-4o-mini",
    generation_task         = "qa",                  # or "preference" for DPO pairs
    hallucination_threshold = 0.7,
    reward_threshold        = 0.7,
    export_formats          = ["alpaca"],
    output_split            = {"train": 0.9, "val": 0.1},
    output_dir              = "output/curated",
)).run()
```

This writes `output/curated/train/sft_alpaca.jsonl` and
`output/curated/val/sft_alpaca.jsonl`, plus the provenance set.

## 2. Publish the dataset

AlignTune's trainers take a dataset name, so load the exported JSONL with the
`datasets` library and push it to the HuggingFace Hub. The auto-generated
`dataset_card.md` is a ready-made README for the dataset repository.

```python
from datasets import load_dataset

ds = load_dataset("json", data_files={
    "train":      "output/curated/train/sft_alpaca.jsonl",
    "validation": "output/curated/val/sft_alpaca.jsonl",
})
ds.push_to_hub("your-org/handbook-qa-curated")
```

## 3. Train

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
    model_name   = "Qwen/Qwen3-0.6B",
    dataset_name = "your-org/handbook-qa-curated",
    backend      = "trl",
    num_epochs   = 3,
    batch_size   = 4,
)
trainer.train()
print(trainer.evaluate())
```

For preference data, export with `generation_task="preference"` and
`export_formats=["dpo"]`, then use AlignTune's RL trainer:

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
    model_name   = "Qwen/Qwen3-0.6B",
    dataset_name = "your-org/handbook-dpo-curated",
    algorithm    = "dpo",
    backend      = "trl",
)
trainer.train()
```

AlignTune's [documentation](https://github.com/Lexsi-Labs/aligntune) covers backend
selection, the other RL algorithms, and evaluation. The provenance manifest from
step 1 stays valid for the published dataset: every training sample traces back to
its source chunk.
