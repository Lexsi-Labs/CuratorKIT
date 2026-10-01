"""Shared handling for LLM-judge failures in HallucinationGate and RewardGate."""

from __future__ import annotations

import warnings

from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

JUDGE_ERROR_MODES = ("reject", "pass")
JUDGE_ERROR_PREFIX = "judge_error:"


def check_on_judge_error(on_judge_error: str) -> str:
    if on_judge_error not in JUDGE_ERROR_MODES:
        raise ValueError(
            f"on_judge_error must be one of {JUDGE_ERROR_MODES}, got {on_judge_error!r}"
        )
    return on_judge_error


def judge_error_result(
    sample: DataSample,
    error: Exception,
    *,
    on_judge_error: str,
    step_name: str,
    step_version: str,
    cfg_hash: str,
    ts,
) -> tuple[DataSample | None, RejectedSample | None]:
    """Fail closed by default: reject the sample and record the judge error.

    With ``on_judge_error="pass"`` the sample passes (legacy behaviour), but the
    error is still recorded in its provenance.
    """
    passed = on_judge_error == "pass"
    record = ProvenanceRecord(
        step_name=step_name, step_version=step_version,
        timestamp=ts, config_hash=cfg_hash,
        notes={
            "error": str(error)[:500],
            "error_type": type(error).__name__,
            "passed_on_error": passed,
        },
    )
    if passed:
        sample.append_provenance(record)
        return sample, None
    rej = RejectedSample(
        **sample.model_dump(),
        rejection_reason=f"{JUDGE_ERROR_PREFIX}{type(error).__name__}",
        rejecting_step=step_name,
    )
    rej.append_provenance(record)
    return None, rej


def warn_judge_errors(step_name: str, passed: list, rejected: list) -> None:
    """Warn once per run when the judge errored, so failures are never silent."""
    n = sum(
        1 for r in rejected if r.rejection_reason.startswith(JUDGE_ERROR_PREFIX)
    ) + sum(
        1 for s in passed
        for p in s.provenance_chain
        if p.step_name == step_name and p.notes.get("passed_on_error")
    )
    if n:
        warnings.warn(
            f"{step_name}: judge failed on {n}/{len(passed) + len(rejected)} samples "
            f"(see provenance 'error' / rejection reason '{JUDGE_ERROR_PREFIX}*').",
            stacklevel=3,
        )
