"""
StratifiedSampler — rebalance skewed task-category distributions.

Over-represented categories are downsampled to their target fraction.
Under-represented categories produce a warning in the manifest — the imbalance
is never silently accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import warnings
from datetime import UTC, datetime

from curatorkit.interfaces import BaseNormalizer
from curatorkit.schema import DataSample, ProvenanceRecord

STEP_VERSION = "0.1.0"


class StratifiedSampler(BaseNormalizer):
    """Downsample over-represented categories; warn on under-represented ones.

    Args:
        category_field: Key in DataSample.metadata (or top-level field) that
            holds the category label. Defaults to 'task_type' (top-level field).
        target_distribution: Mapping of category -> max fraction of total output.
            e.g. {'coding': 0.25, 'general': 0.40}
            Categories not listed are passed through without downsampling.
        seed: Random seed for reproducible sampling.
    """

    def __init__(
        self,
        category_field: str = "source_dataset",
        target_distribution: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        self.category_field = category_field
        self.target_distribution = target_distribution or {}
        self.seed = seed

    def _get_category(self, sample: DataSample) -> str:
        # Check top-level DataSample fields first (task_type, source_uri, etc.)
        val = getattr(sample, self.category_field, None)
        if val is not None:
            return str(val)
        # Fall back to metadata dict
        return str(sample.metadata.get(self.category_field, "unknown"))

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "category_field": self.category_field,
                "target_distribution": self.target_distribution,
                "seed": self.seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        if not self.target_distribution:
            return samples

        # Group by category
        from tqdm import tqdm

        groups: dict[str, list[DataSample]] = {}
        for sample in tqdm(samples, desc="StratifiedSampler", unit="sample"):
            key = self._get_category(sample)
            groups.setdefault(key, []).append(sample)

        # Pass 1 — find binding total from the most constrained category
        binding_total = float("inf")
        shortfalls: dict[str, dict] = {}
        for category, fraction in self.target_distribution.items():
            actual = len(groups.get(category, []))
            if actual == 0:
                warnings.warn(
                    f"[StratifiedSampler] category '{category}' has 0 samples "
                    f"— cannot meet target distribution."
                )
                shortfalls[category] = {
                    "available": 0,
                    "target_fraction": fraction,
                    "reason": "no_samples_available",
                }
                continue
            # How large can the total output be given this category's supply?
            candidate_total = actual / fraction
            binding_total = min(binding_total, candidate_total)

        if binding_total == float("inf"):
            return samples

        # Pass 2 — cap each category at floor(binding_total × fraction)
        rng = random.Random(self.seed)
        result: list[DataSample] = []
        adjustments: dict[str, dict] = {}

        for category, fraction in self.target_distribution.items():
            pool = groups.get(category, [])
            target_count = math.floor(binding_total * fraction)
            kept = rng.sample(pool, min(target_count, len(pool)))
            result.extend(kept)
            adjustments[category] = {
                "original": len(pool),
                "kept": len(kept),
                "removed": len(pool) - len(kept),
                "target_count": target_count,
                "target_fraction": fraction,
            }
            # Record shortfall when the pool was smaller than the target count
            if len(pool) < target_count:
                shortfalls[category] = {
                    "available": len(pool),
                    "target_count": target_count,
                    "target_fraction": fraction,
                    "shortfall": target_count - len(pool),
                    "reason": "pool_smaller_than_target",
                }

        # Pass through categories not in target_distribution unchanged
        for category, pool in groups.items():
            if category not in self.target_distribution:
                result.extend(pool)

        if shortfalls:
            warnings.warn(
                f"[StratifiedSampler] target distribution shortfall in "
                f"{len(shortfalls)} categor{'y' if len(shortfalls) == 1 else 'ies'}: "
                f"{sorted(shortfalls.keys())}. See manifest for details."
            )

        # Append provenance
        ts = datetime.now(UTC).replace(tzinfo=None)
        cfg = self._config_hash()
        for sample in result:
            sample.append_provenance(
                ProvenanceRecord(
                    step_name="StratifiedSampler",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg,
                    notes={
                        "category_field": self.category_field,
                        "binding_total": int(binding_total),
                        "output_total": len(result),
                        "adjustments": adjustments,
                        "shortfalls": shortfalls,
                        "has_shortfall": bool(shortfalls),
                    },
                )
            )

        return result
