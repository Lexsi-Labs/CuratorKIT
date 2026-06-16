"""
DiagnosticProbe — score-conditioned diagnostic for gate rejections.

Probe routing is conditioned on the grounding_score written by the
HallucinationGate to the sample's provenance chain:

  grounding_score >= score_split (default 0.5)
    → near-boundary failure likely
    → Probe 1 (temperature sweep) first, then Probe 2 (prompt variants)

  grounding_score < score_split
    → model fundamentally ignored the source
    → Probe 2a (strict grounding) first, then Probe 1 (temperature sweep),
      then remaining Probe 2 variants

Worst case total: 5 LLM calls per rejected sample (2 temperature sweep + 3 prompt variants).

Context extension (adjacent chunk) is deliberately excluded: both generator
and judge operate on the same fixed context, so cross-chunk probing would
produce coincidental rather than causal diagnoses.
"""

from __future__ import annotations

import copy
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from tqdm import tqdm

from curatorkit.diagnostic.failure_modes import PROMPT_TEMPLATES, FailureDiagnosis, FailureMode
from curatorkit.interfaces import BaseGate
from curatorkit.llm.base import BaseLLM
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

_PROBE_VERSION = "1.0.0"

logger = logging.getLogger(__name__)

# Grounding score below this → parametric probe first; above → temperature sweep first
_DEFAULT_SCORE_SPLIT = 0.5


class DiagnosticProbe:
    """
    Self-consistency diagnostic for gate rejections.

    Attach to a gate: gate.probe = DiagnosticProbe(generator_llm, gate)
    Pipeline calls:   diagnosis = gate.probe.diagnose(rejected_sample)

    Parameters
    ----------
    generator_llm : BaseLLM
        The LLM that produced the failed sample — used for re-generation.
    gate : BaseGate
        The gate that produced the rejection — reused to evaluate re-generations.
    temperatures : list[float]
        Temperature sweep values. Default [0.3, 0.5] (below default generation T of 0.7).
    score_split : float
        Grounding score threshold for probe routing (default 0.5).
        Samples scoring below this go to strict-grounding probe first.
    """

    def __init__(
        self,
        generator_llm: BaseLLM,
        gate: BaseGate,
        temperatures: list[float] | None = None,
        score_split: float = _DEFAULT_SCORE_SPLIT,
        extra_templates: dict[str, str] | None = None,
    ) -> None:
        self.generator_llm = generator_llm
        self.gate = gate
        self.temperatures = temperatures or [0.3, 0.5]
        self.score_split = score_split

        # Merge user-supplied templates with the built-ins; user values take precedence.
        if extra_templates:
            import copy

            self._prompt_templates = {**copy.deepcopy(PROMPT_TEMPLATES), **extra_templates}
        else:
            self._prompt_templates = PROMPT_TEMPLATES

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def diagnose(self, rejected: RejectedSample) -> FailureDiagnosis:
        """Run probes on a rejected sample. Never raises — returns UNKNOWN on error."""
        # DPO pair rejected because the adversarial sample scored too high — refining
        # chosen cannot fix this; only the upstream generator can produce a better pair.
        if "rejected_above_threshold" in (rejected.rejection_reason or ""):
            return FailureDiagnosis.from_mode(
                FailureMode.RESPONSE_QUALITY,
                evidence=[],
                probe_calls=0,
                notes={"skip_reason": "rejected_above_threshold"},
            )
        try:
            return self._run_probes(rejected)
        except Exception as exc:
            logger.warning("DiagnosticProbe error on %s: %s", rejected.id, exc)
            return FailureDiagnosis.from_mode(
                FailureMode.UNKNOWN,
                evidence=[],
                probe_calls=0,
                notes={"error": str(exc)},
            )

    def diagnose_batch(
        self,
        rejected: list[RejectedSample],
        concurrency: int = 32,
    ) -> list[FailureDiagnosis]:
        """
        Diagnose all rejected samples concurrently.

        Runs up to `concurrency` samples in parallel — each sample's probe
        sequence is still sequential internally (each result conditions the
        next probe), but different samples proceed independently. A single
        tqdm bar tracks the batch.
        """
        if not rejected:
            return []

        results: list[FailureDiagnosis | None] = [None] * len(rejected)
        lock = threading.Lock()

        def _run(args: tuple[int, RejectedSample]) -> None:
            idx, sample = args
            diag = self.diagnose(sample)
            with lock:
                results[idx] = diag

        with tqdm(total=len(rejected), desc="DiagnosticProbe", unit="sample") as pbar:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(_run, (i, s)) for i, s in enumerate(rejected)]
                for f in as_completed(futures):
                    f.result()
                    pbar.update(1)

        return results  # type: ignore[return-value]

    # ─────────────────────────────────────────────────────────────────────────
    # Probe sequence
    # ─────────────────────────────────────────────────────────────────────────

    def _run_probes(self, rejected: RejectedSample) -> FailureDiagnosis:
        source_ctx = rejected.input.strip()
        total_calls = 0

        if not source_ctx or not rejected.instruction:
            return FailureDiagnosis.from_mode(
                FailureMode.UNKNOWN,
                evidence=[],
                probe_calls=0,
                notes={"reason": "no_source_context_or_instruction"},
            )

        grounding_score = self._get_grounding_score(rejected)
        low_grounding = grounding_score < self.score_split

        if low_grounding:
            # ── Low-grounding path: strict grounding first ────────────────────
            # Model fundamentally ignored the source → check parametric override
            # before spending 3 calls on a temperature sweep that won't help.
            diag, total_calls, p1_results = self._probe_grounding_first(
                rejected, source_ctx, total_calls
            )
        else:
            # ── Near-boundary path: temperature sweep first ───────────────────
            # Sample nearly passed → instability or marginal score is more likely
            # than a complete parametric override.
            diag, total_calls, p1_results = self._probe_temperature_first(
                rejected, source_ctx, total_calls
            )

        if diag is not None:
            return diag

        # All probes exhausted — classify from provenance heuristic
        mode = self._classify_source_failure(rejected)
        return FailureDiagnosis.from_mode(
            mode,
            evidence=p1_results,
            probe_calls=total_calls,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Routing paths
    # ─────────────────────────────────────────────────────────────────────────

    def _probe_temperature_first(
        self,
        rejected: RejectedSample,
        source_ctx: str,
        total_calls: int,
    ) -> tuple[FailureDiagnosis | None, int, list[bool]]:
        """Temperature sweep → prompt variants."""
        p1_results, p1_samples, total_calls = self._run_temperature_sweep(
            rejected, source_ctx, total_calls
        )
        p1_diag = self._classify_probe1(p1_results, p1_samples, total_calls)
        if p1_diag is not None:
            return p1_diag, total_calls, p1_results

        diag, total_calls = self._run_prompt_variants(rejected, source_ctx, total_calls, p1_results)
        return diag, total_calls, p1_results

    def _probe_grounding_first(
        self,
        rejected: RejectedSample,
        source_ctx: str,
        total_calls: int,
    ) -> tuple[FailureDiagnosis | None, int, list[bool]]:
        """Strict grounding → temperature sweep → remaining prompt variants."""
        p1_results: list[bool] = []

        # Probe 2a first: strict grounding
        regen = self._regenerate(
            rejected,
            source_ctx,
            temperature=self.temperatures[0],
            prompt_template="strict_grounding",
        )
        total_calls += 1
        if regen is not None:
            passed, _ = self.gate.run([regen])
            if passed:
                self._stamp_recovered(
                    passed[0],
                    FailureMode.GENERATOR_PARAMETRIC,
                    "strict_grounding",
                    {"route": "low_grounding", "temperature": self.temperatures[0]},
                )
                return (
                    FailureDiagnosis.from_mode(
                        FailureMode.GENERATOR_PARAMETRIC,
                        evidence=[],
                        probe_calls=total_calls,
                        notes={"passing_probe": "strict_grounding", "route": "low_grounding"},
                        recovered_sample=passed[0],
                    ),
                    total_calls,
                    p1_results,
                )

        # Strict grounding didn't fix it — run temperature sweep
        p1_results, p1_samples, total_calls = self._run_temperature_sweep(
            rejected, source_ctx, total_calls
        )
        p1_diag = self._classify_probe1(p1_results, p1_samples, total_calls)
        if p1_diag is not None:
            return p1_diag, total_calls, p1_results

        # Temperature sweep inconclusive — run remaining prompt variants (skip 2a already done)
        diag, total_calls = self._run_prompt_variants(
            rejected, source_ctx, total_calls, p1_results, skip_strict_grounding=True
        )
        return diag, total_calls, p1_results

    # ─────────────────────────────────────────────────────────────────────────
    # Shared probe runners
    # ─────────────────────────────────────────────────────────────────────────

    def _run_temperature_sweep(
        self,
        rejected: RejectedSample,
        source_ctx: str,
        total_calls: int,
    ) -> tuple[list[bool], list[DataSample | None], int]:
        results: list[bool] = []
        samples: list[DataSample | None] = []
        for t in self.temperatures:
            regen = self._regenerate(rejected, source_ctx, temperature=t)
            total_calls += 1
            if regen is None:
                results.append(False)
                samples.append(None)
                continue
            passed, _ = self.gate.run([regen])
            results.append(bool(passed))
            samples.append(passed[0] if passed else None)
        return results, samples, total_calls

    def _run_prompt_variants(
        self,
        rejected: RejectedSample,
        source_ctx: str,
        total_calls: int,
        p1_results: list[bool],
        skip_strict_grounding: bool = False,
    ) -> tuple[FailureDiagnosis | None, int]:
        """Run strict grounding → domain-specific → instruction regen. First pass wins."""

        # 2a: strict grounding — parametric drift (skip if already run in low-grounding path)
        if not skip_strict_grounding:
            regen = self._regenerate(
                rejected,
                source_ctx,
                temperature=self.temperatures[0],
                prompt_template="strict_grounding",
            )
            total_calls += 1
            if regen is not None:
                passed, _ = self.gate.run([regen])
                if passed:
                    self._stamp_recovered(
                        passed[0],
                        FailureMode.GENERATOR_PARAMETRIC,
                        "strict_grounding",
                        {"temperature": self.temperatures[0]},
                    )
                    return (
                        FailureDiagnosis.from_mode(
                            FailureMode.GENERATOR_PARAMETRIC,
                            evidence=p1_results,
                            probe_calls=total_calls,
                            notes={"passing_probe": "strict_grounding"},
                            recovered_sample=passed[0],
                        ),
                        total_calls,
                    )

        # 2b: domain-specific prompt — domain grounding
        domain_template = rejected.metadata.get("domain_prompt_key", "domain_specific")
        regen = self._regenerate(
            rejected,
            source_ctx,
            temperature=self.temperatures[0],
            prompt_template=domain_template,
        )
        total_calls += 1
        if regen is not None:
            passed, _ = self.gate.run([regen])
            if passed:
                self._stamp_recovered(
                    passed[0],
                    FailureMode.DOMAIN_MISMATCH,
                    domain_template,
                    {"temperature": self.temperatures[0]},
                )
                return (
                    FailureDiagnosis.from_mode(
                        FailureMode.DOMAIN_MISMATCH,
                        evidence=p1_results,
                        probe_calls=total_calls,
                        notes={"passing_probe": domain_template},
                        recovered_sample=passed[0],
                    ),
                    total_calls,
                )

        # 2c: instruction regeneration — poor instruction following
        regen_instr = self._regenerate_instruction(rejected, source_ctx)
        total_calls += 1
        if regen_instr is not None:
            passed, _ = self.gate.run([regen_instr])
            if passed:
                self._stamp_recovered(
                    passed[0],
                    FailureMode.INSTRUCTION_QUALITY,
                    "instruction_regen",
                    {"temperature": self.temperatures[0]},
                )
                return (
                    FailureDiagnosis.from_mode(
                        FailureMode.INSTRUCTION_QUALITY,
                        evidence=p1_results,
                        probe_calls=total_calls,
                        notes={"passing_probe": "regenerated_instruction"},
                        recovered_sample=passed[0],
                    ),
                    total_calls,
                )

        return None, total_calls

    # ─────────────────────────────────────────────────────────────────────────
    # Probe 1 pattern classifier
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_probe1(
        self,
        results: list[bool],
        samples: list[DataSample | None],
        calls_so_far: int,
    ) -> FailureDiagnosis | None:
        """
        Pattern → Diagnosis (first passing sample stored as recovered_sample)
        all False                      → inconclusive (None)
        all True                       → THRESHOLD_MARGINAL
        results[0] and not results[-1] → GENERATOR_TEMPERATURE
        mixed other                    → THRESHOLD_MARGINAL (unstable)
        """
        if not results:
            return None

        n_pass = sum(results)

        if n_pass == 0:
            return None

        first_passer = next((s for s in samples if s is not None), None)

        if n_pass == len(results):
            if first_passer is not None:
                self._stamp_recovered(
                    first_passer,
                    FailureMode.THRESHOLD_MARGINAL,
                    "temperature_sweep",
                    {"pattern": "all_pass"},
                )
            return FailureDiagnosis.from_mode(
                FailureMode.THRESHOLD_MARGINAL,
                evidence=results,
                probe_calls=calls_so_far,
                notes={"pattern": "all_pass"},
                recovered_sample=first_passer,
            )

        if results[0] and not results[-1]:
            if samples[0] is not None:
                self._stamp_recovered(
                    samples[0],
                    FailureMode.GENERATOR_TEMPERATURE,
                    "temperature_sweep",
                    {"pattern": "low_t_pass", "temperature": self.temperatures[0]},
                )
            return FailureDiagnosis.from_mode(
                FailureMode.GENERATOR_TEMPERATURE,
                evidence=results,
                probe_calls=calls_so_far,
                notes={"pattern": "low_t_pass", "suggested_temperature": self.temperatures[0]},
                recovered_sample=samples[0],
            )

        if first_passer is not None:
            t_idx = next(i for i, s in enumerate(samples) if s is not None)
            self._stamp_recovered(
                first_passer,
                FailureMode.THRESHOLD_MARGINAL,
                "temperature_sweep",
                {"pattern": "mixed_unstable", "temperature": self.temperatures[t_idx]},
            )
        return FailureDiagnosis.from_mode(
            FailureMode.THRESHOLD_MARGINAL,
            evidence=results,
            probe_calls=calls_so_far,
            notes={"pattern": "mixed_unstable"},
            recovered_sample=first_passer,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Source failure classifier (fallback — no LLM call)
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_source_failure(self, rejected: RejectedSample) -> FailureMode:
        # Both generator and judge operate on the same provenance-preserved source
        # chunk, so the judge has no independent basis to assess source "thickness".
        # All exhausted-probe fallbacks are classified as SOURCE_AMBIGUOUS (the
        # source/response relationship is unclear but not diagnosable further) or
        # UNKNOWN when no provenance notes are available at all.
        prov_notes: dict = {}
        for rec in reversed(rejected.provenance_chain):
            if "unsupported_claims" in rec.notes:
                prov_notes = rec.notes
                break

        if not prov_notes:
            return FailureMode.UNKNOWN
        return FailureMode.SOURCE_AMBIGUOUS

    # ─────────────────────────────────────────────────────────────────────────
    # Provenance helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_grounding_score(self, rejected: RejectedSample) -> float:
        """Extract grounding_score written by HallucinationGate to provenance."""
        for rec in reversed(rejected.provenance_chain):
            if "grounding_score" in rec.notes:
                return float(rec.notes["grounding_score"])
        return 0.0  # no score → treat as low-grounding

    def _stamp_recovered(
        self,
        sample: DataSample,
        mode: FailureMode,
        probe_path: str,
        extra_notes: dict | None = None,
    ) -> DataSample:
        """Append a DiagnosticProbe provenance record to a recovered sample.

        This seals the provenance chain — a reader can see exactly which probe
        path succeeded and which failure mode was diagnosed.
        """
        notes: dict = {"probe_path": probe_path, "mode": mode.value}
        if extra_notes:
            notes.update(extra_notes)
        sample.append_provenance(
            ProvenanceRecord(
                step_name="DiagnosticProbe",
                step_version=_PROBE_VERSION,
                timestamp=datetime.now(UTC),
                config_hash="",
                notes=notes,
            )
        )
        return sample

    # ─────────────────────────────────────────────────────────────────────────
    # Re-generation
    # ─────────────────────────────────────────────────────────────────────────

    def _regenerate(
        self,
        original: RejectedSample,
        source_context: str,
        temperature: float,
        prompt_template: str = "default",
    ) -> DataSample | None:
        if not source_context:
            return None

        template = self._prompt_templates.get(
            prompt_template, self._prompt_templates.get("default", PROMPT_TEMPLATES["default"])
        )
        prompt = template.format(
            source=source_context,
            question=original.instruction,
        )

        try:
            resp = self.generator_llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=512,
            )
        except Exception as exc:
            logger.debug("Probe re-generation failed T=%.1f: %s", temperature, exc)
            return None

        regenerated_text = resp.text.strip()
        if not regenerated_text:
            # Model returned only thinking tokens or empty output — treat as failure
            logger.debug("Probe re-generation returned empty text T=%.1f", temperature)
            return None
        # For DPO/preference samples the regenerated text becomes the new chosen
        # answer; the original rejected response (adversarial or quality-degraded)
        # is preserved unchanged so the DPO pair structure survives probe recovery.
        return DataSample(
            source_uri=original.source_uri,
            instruction=original.instruction,
            input=source_context,
            output=regenerated_text,
            chosen=regenerated_text if (original.chosen or original.rejected) else "",
            rejected=original.rejected,
            task_type=original.task_type,
            metadata=copy.deepcopy(original.metadata),
            provenance_chain=list(original.provenance_chain),
        )

    def _regenerate_instruction(
        self,
        original: RejectedSample,
        source_context: str,
    ) -> DataSample | None:
        if not source_context:
            return None

        template = PROMPT_TEMPLATES["generate_question"]
        prompt = template.format(source=source_context)

        try:
            resp = self.generator_llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperatures[0],
                max_tokens=128,
            )
        except Exception as exc:
            logger.debug("Probe instruction re-generation failed: %s", exc)
            return None

        new_instruction = resp.text.strip()
        if not new_instruction:
            return None

        # Keep original output/chosen/rejected — only the instruction changes.
        return DataSample(
            source_uri=original.source_uri,
            instruction=new_instruction,
            input=source_context,
            output=original.output,
            chosen=original.chosen,
            rejected=original.rejected,
            task_type=original.task_type,
            metadata=copy.deepcopy(original.metadata),
            provenance_chain=list(original.provenance_chain),
        )
