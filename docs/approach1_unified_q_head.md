# Approach 1: Q-head experiments on new datasets and the unified head

This note covers three things that build on the original ChartQA study: running the same
experiment on three more datasets, a single Q-head that reads the pre-generation and
verifier latents together, and how the swept results get compared.

## 1. Reference: what the ChartQA sweep found

Source: `results/confidence_comparison_report.html` (the original ChartQA report). Qwen3-VL-2B, 2,500 ChartQA test
questions, base accuracy 75.84%. Heads are trained on frozen last-layer states, the VLM
itself is not updated.

| Test metric | Geo-mean token conf. | Pre-gen MLP | Pre-gen linear (sweep winner) | Verifier MLP | Verifier, early stop |
|---|---:|---:|---:|---:|---:|
| Correctness AUROC | 0.750 | 0.863 | 0.895 | 0.887 | 0.891 ± 0.001 |
| Error AUPRC | 0.583 | 0.669 | 0.732 | 0.723 | 0.738 ± 0.005 |
| Brier | 0.178 | 0.123 | 0.109 | 0.112 | 0.111 ± 0.001 |
| ECE | 0.172 | 0.055 | 0.036 | 0.030 | 0.038 ± 0.014 |
| Accuracy at 50% coverage | 92.40% | 95.12% | 97.04% | 96.64% | 97.04% |

Three findings carry over into the design below:

- Token confidence is close to useless as a gate. 1,921 of 2,500 scores were exactly 1.0.
- A linear probe on the prompt latent won the pre-generation sweep, so question difficulty is
  already close to linearly readable before any answer exists.
- The ranking loss never won. Plain BCE was selected for both heads, and early stopping
  helped the verifier a little.

## 2. Three more datasets

The point is to see whether the Q-head result holds outside charts, so each new dataset is a
different kind of input. All three have human-written answers and public labels.

| Key | Source | Domain | Metric | Splits used |
|---|---|---|---|---|
| `textvqa` | `lmms-lab/textvqa` | Reading text in natural photos | VQA soft accuracy, 10 annotators | train sample, val and test carved from the labeled validation split by image |
| `docvqa` | `HuggingFaceM4/DocumentVQA` | Scanned industry documents | ANLS similarity | train sample, val and test carved from the labeled validation split by document |
| `scienceqa_img` | `derek-thomas/ScienceQA` | Science diagrams, multiple choice | Option letter match | official train, validation, test, image questions only |

Defaults are 8,000 train, 1,500 val, 2,500 test (ScienceQA is smaller, so it uses what the
image subset has). Change them with `TRAIN_SIZE`, `VAL_SIZE`, `TEST_SIZE`.

### Why these and not others

- **Hidden test labels.** TextVQA and DocVQA test answers are empty on the Hub, so the held-out
  data is split from validation. The split is grouped by image or document, otherwise the same
  page could sit in both val and test and inflate the numbers.
- **Binary labels that mean something.** The Q-head needs a clean correct or wrong target. The
  cutoffs live in `scoring.py`: VQA score of 0.5 or more (at least two of ten annotators
  agree), ANLS similarity of 0.9 or more, exact option letter.
- **Different from charts.** Natural scenes with text, dense documents, and science diagrams
  stress different parts of the vision encoder than plotted data.
- **Rejected:**
  - GQA: questions and images live in separate configs and need a join, and a small model's
    accuracy is low enough that the error class dominates.
  - VQAv2: only validation is labeled on the lmms-lab copy, and it is known for noisy answers.
  - OK-VQA: 5k questions, and answers depend on outside knowledge the image does not contain.
  - AI2D, MathVista: test-only or about 1k labeled rows, too small to train a head on.
  - InfographicVQA: only a labeled validation split.

DocVQA page images are large, so `build_qhead_splits.py` caps the longest side at 1,536 px
(`MAX_SIDE`). The cap is applied once at build time, so baseline generation and feature
extraction see identical pixels.

## 3. The unified Q-head

The original setup trains two heads on different cached states: a pre-generation head on the
last prompt token, and a verifier on the last token of the prompt plus the generated answer.
The unified head takes both.

```
h_prompt (d) --LayerNorm--+
                          +-- concat [ z_prompt | z_answer | flag ] -- linear or MLP -- logit
h_answer (d) --LayerNorm--+
```

- Each half gets its own LayerNorm so the two latents are on the same scale before fusing.
- `flag` is 1 when an answer latent is present and 0 when it is not.
- With no answer, the answer half is zeros and the flag is 0. This is **pre-generation mode**.
  The same weights give a score before generation, so the gate can run first and the verifier
  score is only needed if a correction runs.
- With an answer it is **joint mode**, the verifier.

### How the loss is handled

Both modes predict the same label (was the generated answer correct), so the targets never
conflict. The risk is different: the answer latent is the stronger signal, and a head trained
only on joint inputs can lean on it and ignore the prompt half. Then pre-generation mode, which
sees zeros for the answer, would be poor. The loss is built to prevent that:

```
L = BCE(f(p, g), y)                              joint mode
  + pre_weight  * BCE(f(p, none), y)             pre-generation mode, same weights
  + rank_weight * ranking(f(p, g), y)            correct above wrong, off by default
  + pair_weight * softplus(f(p,g) - f(p,r))      wrong rows only, reference answer above candidate
```

- Both BCE terms are proper scoring rules, so calibration comes from the loss itself. A
  per-mode temperature fitted on validation only fixes leftover scale.
- Both modes are computed on every batch. There is no random dropping of the answer half, so
  there is no masking variance and the two terms keep a fixed ratio.
- `pre_weight` is swept over {0, 0.5, 1.0}. At 0 the head is trained as a plain concat
  verifier, so comparing it with the full loss is a direct ablation of the pre term.
- `pair_weight` reuses the reference-answer contrast from the original verifier head. `rank_weight` stays 0 because the
  ChartQA sweep never selected it. It is a flag if someone wants to retest it.
- Weights are fixed hyperparameters picked on validation, not learned. With a few thousand
  rows a learned weighting would mostly fit noise.

### Selection and what gets reported

`sweep_unified_q_head.py` follows the early-stop sweep: 3 seeds per config, best epoch per run
chosen by validation error AUPRC then AUROC, winner picked by the mean over seeds, test touched
once. The default `--select mean` averages the validation score of both modes, since the claim
is that one set of weights serves both. `--select joint` or `--select pre_generation` optimize
one mode only.

Grid: architecture {linear, mlp_256, mlp_512} x learning rate {3e-4, 1e-3} x pre_weight
{0, 0.5, 1.0} x pair_weight {0, 0.25}, so 36 configs and 108 trainings.

### What would count as a win

The unified head earns its place if, per dataset, one of these holds on test:

1. Its pre-generation mode is at least as good as the separate pre-gen head, so fusing did not
   cost anything on the pre-answer signal.
2. Its joint mode beats the separate verifier by more than the seed spread.

If neither holds, the separate heads stay and the unified head is dropped.

## 4. Running it

Set up the environment once (`sbatch slurm/setup_environment.sbatch`), then per dataset:

```
bash slurm/run_qhead_pipeline.sh textvqa
bash slurm/run_qhead_pipeline.sh docvqa
bash slurm/run_qhead_pipeline.sh scienceqa_img
bash slurm/run_qhead_pipeline.sh chartqa_full     # same chain on the original ChartQA dataset
```

Each call submits a dependency chain: build splits, then baseline and feature extraction for
train, val, and test, then the separate-head sweep and the unified sweep. When they finish:

```
python -m experiments.approach1_latent_self_correction.compare_q_heads_report
```

This writes `results/q_head_comparison.md` with five scorers side by side per dataset: token
confidence, separate pre-gen head, separate verifier, unified pre-gen mode, unified joint mode.
Older ChartQA outputs at `q_models/sweep_results.json` are picked up as a fallback.

Offline checks that need no GPU or downloads:

```
python -m tests.test_config
python -m tests.test_scoring_and_splits
python -m tests.test_unified_q_head
```

## 5. Status

Done: dataset selection and verification against the Hub, split builder, scoring, baseline
hook, unified head and sweep, analysis tooling, comparison report, Slurm chain, and the real
runs of TextVQA, DocVQA, and ScienceQA on the ICE cluster (Qwen3-VL-2B, one L40S per job).
Offline tests cover the scoring rules, the grouped split, and the unified head and sweep.

Results are written up with charts, intervals, ablations, and limitations in
`results/q_head_multidataset_report.html`. In short, on test:

| Dataset | Base accuracy | Token confidence AUROC | Best separate head AUROC | Unified joint AUROC |
|---|---:|---:|---:|---:|
| TextVQA | 81.0% | 0.726 | 0.868 | 0.867 |
| DocVQA | 87.0% | 0.685 | 0.788 | 0.811 |
| ScienceQA | 86.7% | 0.654 | 0.918 | 0.925 |

By the win criteria in Section 3: the unified head's joint mode beats the separate verifier on
DocVQA and ScienceQA (paired bootstrap interval excludes zero) and ties it on TextVQA. Its
pre-generation mode is within 0.006 AUROC of the dedicated pre-generation head on every
dataset, slightly below on TextVQA and DocVQA and slightly above on ScienceQA. So criterion 2
holds on two of three datasets and criterion 1 holds on one, with small deficits elsewhere.
Taken together, the unified head replaces two models with one at little or no cost.

Still open:

- The correction step that acts on a flagged answer. The counterfactual control (does the score
  move when the visual evidence is corrupted) is now part of the mixture experiments below.
- The Qwen3-VL 4B check.
- A caveat on the data: TextVQA and DocVQA val and test are carved from the labeled
  validation split, grouped by image or document, because the official test answers are hidden.

## 6. Dataset mixture and the new loss

The per-dataset results above train and test a head on one dataset at a time. The next set of
experiments asks what happens when one head has to serve all four datasets, and tests a new
loss that is meant to make the score track visual grounding.

### The mixture

The mixture is the union of the four datasets' existing splits: its train split is the four
train splits pooled, and the same for validation and test. Every row keeps a dataset id so
results are reported pooled and per dataset. The calibration temperature is fitted once on the
pooled validation split, which is how a single deployed head would be calibrated.

### Three experiments

1. **The model's own confidence and linear probes on the mixture.** Token confidence needs no
   training. The linear probes are the linear-architecture heads of the sweeps below.
2. **Separate and unified heads on the mixture, old loss.** The pre-generation head, the
   verifier head, and the unified head, trained with the loss used so far.
3. **The same heads with the new loss.**

### The new loss

```
P_theta(S_confidence | h_answer, h_question)         the head itself
L_contrastive(h_predicted, h_correct)                 reference answer scores above a wrong prediction
L_contrastive(h_clean, h_corrupt) -> S_confidence     score drops when the image is corrupted
```

The first line is the existing correctness head trained with cross-entropy. The second term is
the pairwise reference-answer term already in the old loss, kept here as one of the two
contrastive terms. The third term is new. It needs the hidden state of the same question and
the same answer under a corrupted image, which is cached by `extract_corrupt_features.py`.
Corruption replaces the image with a different example's image from the same dataset and split,
so the question and the answer text are unchanged but the evidence is gone. The term is a
margin-free pairwise loss, `softplus(S_corrupt - S_clean)`, applied on examples the model
answered correctly, because a wrong answer has no grounding to lose.

Read as a loss family, **old** means the grounding weight is 0 and **new** means it is above 0.
The sweeps cover the two contrastive terms separately, so the contribution of each can be read
off:

| Pair term (predicted vs correct) | Grounding term | Cell |
|---|---|---|
| off | off | plain |
| on | off | contrast only |
| off | on | grounding only |
| on | on | the new loss as specified |

For the unified head the grounding term is applied to both modes, joint and pre-generation.

### What is measured

On top of the usual metrics, pooled and per dataset, the experiments measure grounding
directly: for each test answer the score on the clean input against the score on the corrupted
input. Two numbers are reported, the share of correct answers whose score drops (order
accuracy, 0.5 is no effect) and the mean drop in predicted probability. The model's own
confidence is measured the same way, teacher-forced on the same answer text under the clean and
the corrupted image, so the comparison is like for like.

### Protocol

Each configuration is trained on three seeds with early stopping on validation, the winner is
the configuration with the best mean validation score, and test numbers are read for winners
only: the best of each loss family, the best linear probe of each family, and the four cells of
the table above. Grids: separate pre-generation head (architecture, learning rate, ranking
weight, grounding weight), separate verifier (architecture, learning rate, pair weight,
grounding weight), unified head (architecture, learning rate, pre-generation weight, pair
weight, grounding weight), 36 to 72 configurations each, 432 trainings in total.

### Interpretation of the loss

The image of the loss was read as follows, and it is worth confirming: the second contrastive
term is implemented at the score level, as a pairwise loss between the scores of the correct
and the predicted answer, not as a contrast between embeddings. If an embedding-level contrast
was intended, `head_loss` in `mixture_heads.py` is the one place to change.

### Sequential training

The mixture trains on all datasets at once. The harder question is whether a head that has
been trained on dataset A still scores A well after it has been trained on B and then C. The
sequential experiments (`continual.py`, `sweep_continual.py`) train a head on one dataset at a
time, in three orders (forward, reverse, shuffled), with early stopping on the current
dataset's validation split, and score every dataset's test split after every stage. For each
head kind, the head configuration of each loss family is the winner of the mixture sweep, and
each variant is run on three seeds, with and without a replay buffer of 10% of every earlier
dataset's training rows.

Metrics, from the stage-by-dataset matrix of AUROC: final average, forgetting (the best score a
dataset had after its own stage minus its final score, averaged over the datasets trained
before the last one), backward transfer (final minus score right after its own stage), and
retention (final minus the score of a head trained on that dataset alone). The grounding check
is tracked the same way.

### Results

The experiments above were run on the ICE cluster (Qwen3-VL-2B, one L40S per GPU job). The full
write-up, with charts, intervals, ablations, and limits, is in
`results/approach1/05_mixture_new_loss_report.html`. In short, on the pooled test split:

- **Mixture.** The untrained answer-token probability has AUROC 0.719. The unified head trained on the
  mixture reaches 0.881 (old loss) and 0.882 (new loss). Per dataset it is within 0.015 AUROC of
  the head trained on that dataset alone, and above it on ChartQA, TextVQA, and DocVQA.
- **Sequential training.** Trained through all four datasets in sequence, heads forget little:
  0.011 AUROC for the unified head, 0.013 for the pre-generation head, 0.032 for the verifier.
  A replay buffer of 10% of earlier training rows cuts the verifier's forgetting to 0.005 and
  changes nothing for the others. The new loss does not change forgetting.
- **The new loss.** The grounding term alone costs no AUROC and raises the share of correct
  answers whose score falls under a corrupted image (unified head: 93% to 96%). The pair term
  alone lowers AUROC by 0.01 to 0.02, so the loss with both terms is slightly worse than the old
  loss. Grounding persists through sequential training under both losses, and is higher with the
  new loss at every stage.
- **The answer-token probability** already falls for 95% of correct answers under a corrupted
  image. Its weakness is ranking, not noticing that the evidence is gone.

### Corrupted image instead of a swapped one

The grounding term of the new loss needs a corrupted state h_corrupt for the same question and
answer. The first version took it from another example's image (swap). A head can meet that term
by noticing that the question and image no longer match, which says little about whether the
image supports the answer. The second version keeps the right image and mixes Gaussian noise into
it, x' = sqrt(a) x + sqrt(1 - a) eps with a = 0.15. The loss is the same,

L = L_cls + w_pre L_pre + w_pair L_pair + lambda L_ground,

L_ground = mean over correct answers of softplus(margin - (s(x_clean) - s(x_corrupt))),

and only the source of x_corrupt changes. Mask (60% of a 16 by 16 grid blanked) and shuffle (4 by
4 patches permuted) are extracted for the test splits only and are used to grade heads on
corruptions they never trained on. The noise strength was picked from a 30-example smoke test.

Experiments, all with three seeds and the same selection protocol as the swap version:
the pooled mixture (`mixture_noise`), the swap-trained heads graded again on noise, mask and
shuffle (`mixture_xeval`), each dataset alone with swap and with noise, and sequential training
on the noise mixture. Results are in
`results/approach1/06_corruption_new_loss_report.html`. In short, pooled AUROC is unchanged
(within 0.0024), the share of correct answers whose score falls under noise rises from 0.92 to
0.94 (old loss) to 0.97 to 1.00 (noise loss), and heads trained with one corruption also improve
on the others. High order accuracy is partly easy, since a noised image moves the hidden state a
lot, so a corruption that removes only the evidence for the answer would be a stronger test.
