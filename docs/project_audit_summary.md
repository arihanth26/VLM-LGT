# Project / Audit Summary

Fall 2026 · Submitted with the code/audit freeze packet

## 1. Project claim and contribution

Small vision-language models (VLMs) are inexpensive but often answer visual questions confidently and incorrectly. Applying a larger model or an expensive correction pipeline to every question can improve accuracy, but removes much of the efficiency advantage of the small model. This project studies a cost-aware alternative: estimate confidence for each question, then allocate additional inference compute only where it is expected to help.

Our primary claim is that confidence-guided routing improves the accuracy obtained under a fixed inference-compute budget. The routing policy may keep the initial Qwen3-VL-2B answer, generate additional 2B candidates, invoke structured visual or reasoning corrections, or escalate to Qwen3-VL-4B/8B. A learned ranker selects among the resulting candidates. We evaluate the complete system as an accuracy–compute frontier rather than treating its highest-compute configuration as the intended deployment policy.

Compute is reported as average **2B-equivalent model passes per question**. One 2B generation or scoring invocation counts as one pass; 4B and 8B generations count as two and four pass-equivalents, respectively. This is a reproducible proxy based on model scale, not a claim about wall-clock latency.

Relative to uniformly applying more inference, simple retry, best-of-*k*, voting, or random routing at the same budget, the contribution is a unified decision system that combines:

1. multiple confidence estimators with different predictive quality and inference costs;
2. validation-selected routing thresholds under explicit compute budgets;
3. several possible uses of additional compute, from 2B reasoning to larger-model escalation; and
4. learned candidate selection using confidence, source provenance, and cross-source agreement.

The maximum-compute system serves as an empirical reference ceiling. The main research object is the frontier: how much of that performance can be recovered at lower average cost.

## 2. Method / system description

The input is an image and question. The initial output is the greedy answer from frozen Qwen3-VL-2B. The system then follows four stages.

**Confidence estimation.** We compare token probability from generation; pre-generation correctness heads and linear probes on the question/image representation; post-answer heads and probes; and the model's prompted yes/no judgment of its answer. These gates have different costs. Token confidence and the pre-generation scorers require no additional model pass after their representations are available, while post-answer encoding or a judge invocation adds a pass. Consequently, the best correctness predictor need not be the best routing policy under a compute constraint.

**Budgeted routing.** For each confidence signal and target routing share, a threshold is fixed using the validation split. On test, every question is routed independently using that fixed threshold. Questions not routed retain the original greedy answer. We compare against random routing at the same budget.

**Candidate generation and escalation.** Routed questions can receive temperature samples, contrastive-decoding answers, answers from zoomed or grounded crops, a deterministic step-by-step answer, three sampled reasoning answers, and short or step-by-step Qwen3-VL-8B answers. Separate experiments also route directly to Qwen3-VL-4B or 8B. The original greedy answer remains in the candidate pool, so routing does not force the system to replace it.

**Candidate ranking.** Each distinct candidate is encoded by the 2B model in the full-image context. Ranker v3 scores candidates using scalar confidence features, candidate-source indicators, source agreement, and dataset identity. The cost-aware version omits the model's yes/no judge, saving one scoring pass per candidate. The ranker is trained with a within-question softmax objective so that correct candidates score above incorrect candidates from the same question.

Qwen3-VL checkpoints and their standard generation behavior are off-the-shelf and remain frozen. Our team implemented the dataset pipeline, confidence heads and probes, correction prompts and candidate sources, routing policies, rankers, cost accounting, statistical evaluation, reports, and Slurm pipelines.

## 3. Experiments and evaluation

The completed evaluation contains 9,517 test questions pooled from ChartQA, TextVQA, DocVQA, and the image subset of ScienceQA. Rankers train on 12,000 questions (3,000 per dataset); model and routing choices are selected on 6,420 validation questions, and test is then read once.

The primary metric is pooled task accuracy: relaxed numeric accuracy for ChartQA, VQA accuracy for TextVQA, ANLS-based correctness for DocVQA, and option correctness for ScienceQA. We report the difference from the same questions' 2B greedy answers, paired 95% bootstrap intervals, exact McNemar tests, and fixed-versus-broken answer counts. Routing experiments additionally report average 2B-equivalent passes and improvement over random routing at matched budget.

The central comparisons are:

- greedy Qwen3-VL-2B;
- uniformly running 4B or 8B;
- random routing at the same correction share;
- token, head, probe, and judge gates;
- prompted retry, best-of-*k*, voting, and confidence-based candidate selection;
- learned rankers over identical candidate pools; and
- full correction versus validation-selected policies capped at 2, 3, or 5 average passes.

Controls and ablations include independent ranker retraining, removal of dataset identity, confidence inputs, source agreement, the Q-head, linear probe, token features, and the judge; matched-cost routing baselines; per-dataset results; candidate-source accuracy and oracle headroom; and accuracy-versus-cost curves. All reported configurations and thresholds are selected on validation rather than test.

## 4. Main results so far

The greedy 2B baseline is **82.78%** accurate. The completed unified frontier is:

| Policy selected on validation | Avg. 2B-equivalent passes | Test accuracy | Gain over greedy |
|---|---:|---:|---:|
| Greedy 2B | 1.00 | 82.78% | — |
| Best policy below 2 passes | 1.79 | 86.32% | +3.54 points |
| Best policy below 3 passes | 2.97 | 87.80% | +5.02 points |
| Best policy below 5 passes | 4.28 | 88.27% | +5.50 points |
| Full candidate system, 40% confidence gate | 8.26 | 89.06% | +6.28 points |
| Full candidate system, 50% confidence gate | 9.94 | 89.31% | +6.54 points |
| Full candidate system on every question | 18.27 | 89.53% | +6.76 points |

The 50% gate therefore retains 6.54 of the full system's 6.76-point gain while applying the expensive correction pool to 49.7% of test questions. It fixes 760 initially wrong answers and breaks 138 initially correct answers. Its improvement over randomly routing the same share is +3.09 points.

The frontier also shows that additional small-model inference is not always the best use of compute. Running 4B on every question reaches **85.85%** at a nominal cost of 2 pass-equivalents. Selectively routing about 30% of questions to 4B reaches **86.03%** at 1.59 passes, while the best 2B-only policy below 2 passes reaches **85.51%** at 1.99 passes. Selective escalation can therefore dominate continued 2B candidate generation under this cost model.

Confidence quality and routing value are not identical. Learned heads and probes generally predict correctness better than raw token confidence, but token confidence often wins a budgeted routing decision because it is available without an additional model pass. Which gate is preferable depends on whether the objective is correctness prediction, identifying questions a particular correction can fix, or maximizing final accuracy after charging for the gate itself.

Representative failures clarify why ranking and routing are necessary. Prompting every answer to retry reduces accuracy; deeper correction applied indiscriminately can produce format errors or replace correct answers; and the correctness head alone is not a reliable within-question candidate selector. Ranker ablations show that cross-source agreement is especially important, while removing the judge has little cost and enables cheaper candidate scoring. The high-compute reference still remains below the candidate oracle, so candidate selection and candidate coverage both leave headroom.

Known limitations are that pass-equivalent costs are a normalized proxy rather than measured end-to-end latency; results are concentrated on one model family and four trained-on datasets; the ranker uses supervised correctness labels; and the largest gains use 8B candidates. Unfinished transfer, alternative-family, and additional-base-model experiments are intentionally excluded from this summary.

## 5. Reproducibility pointers

The audit-freeze branch and tag will be created after this summary is finalized: **[FREEZE BRANCH/TAG AND COMMIT TO BE INSERTED]**.

The correction pipeline is documented in `docs/pipeline.md`. The principal completed reports are:

- `results/approach1/11_correction_ranker_report.html`: rankers v1–v2;
- `results/approach1/12_combined_ranker_report.html`: Ranker v3, sampled reasoning, and gating;
- `results/approach1/13_escalation_report.html`: larger-model routing, escalation candidates, and the unified frontier.

Machine-readable evidence is under `results/approach1/04_sweep_data/`, especially:

- `candidate_ranker_v3*.json` for ranker results and ablations;
- `gating_v3*.json` for validation thresholds and accuracy–cost results;
- `escalation.json` and `cascade.json` for 4B/8B routing;
- `baselines.json` for matched-candidate baselines; and
- `latency.json` for the separate component-timing study, which is not used as the primary cost axis here.

The main execution entry points are:

```bash
bash slurm/run_correction_v3_pipeline.sh
bash slurm/run_escalation_pipeline.sh
bash slurm/run_escalation_ranker_pipeline.sh
bash slurm/run_reasoning_8b_pipeline.sh
bash slurm/run_next_steps_pipeline.sh
```

Individual stages use `slurm/train_ranker_v3.sbatch`, `slurm/gate_ranker_v3.sbatch`, `slurm/analyze_escalation.sbatch`, and `slurm/compute_baselines.sbatch`. Trained heads and probes are under `trained_heads/approach1/`; saved ranker checkpoints and large raw candidate pools are working artifacts under `experiments/approach1_latent_self_correction/results/`. Reproducing candidate generation requires GPUs and access to Qwen3-VL-2B/4B/8B checkpoints. The committed JSON and HTML reports allow the completed statistical analyses and claims to be audited without regenerating every model answer.

## 6. Claims for reviewers to check

1. **Budgeted improvement:** validation-selected confidence routing reaches 86.32%, 87.80%, and 88.27% under average budgets of 2, 3, and 5 pass-equivalents, respectively, from an 82.78% greedy baseline.
2. **Recovery of the reference gain:** a 50% token-confidence gate reaches 89.31% at 9.94 passes, compared with 89.53% at 18.27 passes when the full correction pool is applied universally.
3. **Routing is not random:** the 50% full-system gate exceeds random routing at the same budget by 3.09 points, with thresholds fixed on validation.
4. **Compute should be allocated across model sizes:** selective 4B routing reaches 86.03% at 1.59 pass-equivalents and outperforms both universal 4B at 85.85% and the best 2B-only policy below two passes at 85.51%.
5. **Confidence has decision-dependent value:** learned heads/probes are stronger correctness predictors, but a zero-extra-pass token-confidence gate is often selected when final accuracy is optimized subject to compute cost.
6. **Candidate selection is essential:** learned within-question ranking materially outperforms prompted retry and simple candidate selection; source-agreement ablation causes the largest reported ranker-input decline, while omitting the judge preserves accuracy and lowers scoring cost.

## Contributor statement

- **Aryan Roy:** implemented the full-image baseline, token-confidence evaluation, correctness heads, the original ChartQA study, and the first confidence-guided correction experiments.
- **[Contributor 2 — name]:** [specific implemented components and experiments].
- **[Contributor 3 — name]:** [specific implemented components and experiments].
- **[Contributor 4 — name]:** [specific implemented components and experiments].

