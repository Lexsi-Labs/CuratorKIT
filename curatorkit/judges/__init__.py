"""
Opt-in LLM-as-judge rubrics.

  prometheus.py — Prometheus-2 absolute-grading judge (Kim et al., 2024).
                  https://arxiv.org/abs/2405.01535

Public API:
  from curatorkit.judges import (
      PrometheusJudge,
      PrometheusRewardGate,
      ABSOLUTE_GRADING_RUBRICS,
  )
"""

from curatorkit.judges.deepeval import (
    DeepEvalJudge,
    DeepEvalRewardGate,
    GEvalResult,
)
from curatorkit.judges.factscore import (
    AtomicFact,
    FActScoreGate,
    FActScoreJudge,
    FActScoreResult,
)
from curatorkit.judges.prometheus import (
    ABSOLUTE_GRADING_RUBRICS,
    PrometheusJudge,
    PrometheusRewardGate,
    ScoreRubric,
)

__all__ = [
    "PrometheusJudge",
    "PrometheusRewardGate",
    "ScoreRubric",
    "ABSOLUTE_GRADING_RUBRICS",
    "DeepEvalJudge",
    "DeepEvalRewardGate",
    "GEvalResult",
    "FActScoreJudge",
    "FActScoreGate",
    "FActScoreResult",
    "AtomicFact",
]
