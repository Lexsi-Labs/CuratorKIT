# Method references

Which published method each generation task, judge and gate implements. "Original"
means a CuratorKIT design with no single external reference.

## Generation tasks

| Class | `generation_task` | Reference |
|---|---|---|
| `QAGenerationTask` | `qa` | Source2Synth-style grounded QA |
| `EvolInstructTask` | `evol` | WizardLM Evol-Instruct ([2304.12244](https://arxiv.org/abs/2304.12244)) |
| `PreferenceGenerationTask` | `preference` | UltraFeedback-style pairs ([2310.01377](https://arxiv.org/abs/2310.01377)) |
| `GRPORolloutTask` | `grpo` | DeepSeekMath GRPO rollouts ([2402.03300](https://arxiv.org/abs/2402.03300)) |
| `ChainOfThoughtTask` | `cot` | Chain-of-thought ([2201.11903](https://arxiv.org/abs/2201.11903)) |
| `MagpieTask` | `magpie` | Magpie, ICLR 2025 ([2406.08464](https://arxiv.org/abs/2406.08464)): pre-query template harvest, then a response pass |
| `PersonaDrivenTask` | `persona` | Persona Hub ([2406.20094](https://arxiv.org/abs/2406.20094)), the paper's prompt templates |
| `SelfRewardingTask` | `self_rewarding` | Self-Rewarding LMs, ICML 2024 ([2401.10020](https://arxiv.org/abs/2401.10020)): N candidates, K judge samples, additive 5-point rubric |
| `ConstitutionalRevisionTask` | `constitutional` | Constitutional AI ([2212.08073](https://arxiv.org/abs/2212.08073)): critique and revise |
| `ConstitutionalPreferenceTask` | `constitutional_preference` | Same paper: (revised, original) DPO pairs |
| `BestOfNTask` | `best_of_n` | Best-of-N with a pluggable scorer ([2009.01325](https://arxiv.org/abs/2009.01325), Tulu 3 [2411.15124](https://arxiv.org/abs/2411.15124)) |
| `JointBundleTask` | `joint_bundle` | Original: one pass emits linked SFT + DPO + GRPO samples (shared `bundle_id`) |
| `MultiHopQATask` | `kg_multihop` | GraphGen ([2505.20416](https://arxiv.org/abs/2505.20416)): atomic / aggregated / multi-hop QA over graph paths |
| `CoverageDirectedGenerator` | — | Original: steers the next batch toward thin embedding clusters |

## Judges and gates

| Class | Reference |
|---|---|
| `HallucinationGate` | Grounding score against the source chunk |
| `RewardGate` | UltraFeedback-style multi-dimension judge |
| `PrometheusJudge`, `PrometheusRewardGate` | Prometheus 2 ([2405.01535](https://arxiv.org/abs/2405.01535)): 1-5 absolute grading |
| `DeepEvalJudge`, `DeepEvalRewardGate` | G-Eval, NAACL 2023 ([2303.16634](https://arxiv.org/abs/2303.16634)) |
| `FActScoreJudge`, `FActScoreGate` | FActScore, EMNLP 2023 ([2305.14251](https://arxiv.org/abs/2305.14251)): atomic-fact decompose and verify |
| `DecontaminationGate` | GPT-3 §4 n-gram overlap protocol ([2005.14165](https://arxiv.org/abs/2005.14165)); `n` configurable (13 default, Llama 3 used 8) |
| `PIIGate` | Microsoft Presidio, regex fallback |
| `JailbreakGate` | Meta Llama Prompt Guard 2, keyword fallback |
| `OutputSafetyGate` | Llama Guard hazard taxonomy (S1-S14), regex fallback |

## Knowledge graph, coverage, cost

| Class | Reference |
|---|---|
| `KnowledgeGraph`, `KGExtractor` | GraphGen triple extraction; in-memory store, no graph library |
| `GraphCoverageAnalyser` | Graph-structural coverage in place of GraphGen's trainee-model ECE |
| `CoverageAnalyser`, `DiversityMonitor` | Spherical k-means over embeddings; distinct-n collapse warning |
| `CostGateway`, `TokenBudget`, `CostReport` | LiteLLM `completion_cost`, per-stage attribution, hard/soft caps |
| `BillOfMaterials` | SHA-256 of every output file plus a self-signature; `verify()` re-hashes |
| `DryRunCostEstimator` | Pre-run token and USD estimate from the step list |
