"""
Cost controls and bill of materials.

Components:
  gateway.py — CostGateway: wraps a BaseLLM, tracks tokens + cost per call.
  budget.py  — TokenBudget: hard / soft caps with BudgetExceeded exception.
               When the cap fires, the pipeline halts gracefully and exports
               what was produced (no partial-output loss).
  report.py  — CostReport: per-stage attribution + summary.
  bom.py     — BillOfMaterials: cryptographic verification of every output
               file + manifest. The BoM is itself checksummed.

Public API:
  from curatorkit.cost import (
      CostGateway,
      TokenBudget, BudgetExceeded,
      CostReport,
      BillOfMaterials,
  )
"""

from curatorkit.cost.bom import BillOfMaterials
from curatorkit.cost.budget import BudgetExceeded, TokenBudget
from curatorkit.cost.estimator import CostEstimate, DryRunCostEstimator, StageEstimate
from curatorkit.cost.gateway import CostGateway
from curatorkit.cost.report import CostReport

__all__ = [
    "CostGateway",
    "TokenBudget",
    "BudgetExceeded",
    "CostReport",
    "BillOfMaterials",
    "DryRunCostEstimator",
    "CostEstimate",
    "StageEstimate",
]
