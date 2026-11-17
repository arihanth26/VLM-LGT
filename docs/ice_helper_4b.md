# Running part of the 4B study from a second ICE account

This runs the **Qwen3-VL-4B candidate chains for DocVQA and ScienceQA** (12 GPU jobs) from your own ICE account, in
parallel with the jobs already running on `ajayavijayan3`. Each ICE user has their own GPU budget, so this roughly
halves the time left for the 4B study. The results must be in by **Thursday end of day**.

You only run GPU jobs and share the output folder. Nothing is trained or evaluated on your side: the ranker and the
gating run on `ajayavijayan3` once every chain's output is gathered there.

## What the jobs do

The 4B study repeats the whole 2B correction pipeline with Qwen3-VL-4B as the base model, on exactly the same
questions. Each job is one shard (half) of one dataset split. It runs these stages in order (`slurm/run_base_model_chain.sbatch`):

| Stage | Module | Output (in your copy) |
|---|---|---|
| Question embeddings with 4B | `extract_prompt_features` | `experiments/approach1_latent_self_correction/q_features/<dataset>_<split>_4b.pt` |
| Greedy answer + 10 samples, every answer scored | `candidate_pools` | `results/base_4b/pools/T1/<split>/` |
| Round 1: contrastive decoding, zoomed crops | `correction_candidates` | `results/base_4b/correction/<split>/` |
| Round 2: grounded crops, reasoning, judge | `correction_v2` | `results/base_4b/correction_v2/<split>/` |
| Round 3: 3 sampled reasoning answers | `correction_v3_cot` | `results/base_4b/correction_v3/<split>/` |

(`results/` means `experiments/approach1_latent_self_correction/results/`.) Every stage is resumable: if a job stops,
submitting it again continues where it left off. Expect 3–5 hours per job for ScienceQA and 4–8 hours for DocVQA.

## Where everything is

| What | Path |
|---|---|
| Code | `https://github.com/arihanth26/VLM-LGT` (branch `main`) |
| Main working copy (read access granted to `mjadhav6`) | `/storage/ice1/9/7/ajayavijayan3/VLM-LGT` |
| Python environment (reused, read-only) | `/storage/ice1/9/7/ajayavijayan3/VLM-LGT/.venv` |
| Datasets (reused, read-only, 25 GB) | `/storage/ice1/9/7/ajayavijayan3/VLM-LGT/data/approach1_latent_self_correction/raw` |
| 2B round-2 outputs (only their question lists are read; 161 MB, copied) | `/storage/ice1/9/7/ajayavijayan3/VLM-LGT/experiments/approach1_latent_self_correction/results/correction_v2` |
| Your copy | `~/scratch/VLM-LGT` (keep it in scratch, not home: home is small) |

## Steps

Everything runs on the ICE login node, except the jobs themselves.

**1. Log in and get the code.**

```
ssh mjadhav6@login-ice.pace.gatech.edu
cd ~/scratch
git clone https://github.com/arihanth26/VLM-LGT.git
cd VLM-LGT
```

**2. Link the shared environment and data, and copy the 2B question lists.**

```
O=/storage/ice1/9/7/ajayavijayan3/VLM-LGT
E=experiments/approach1_latent_self_correction
ln -s $O/.venv .venv
rm -r data/approach1_latent_self_correction/raw   # only holds an empty .gitkeep in a fresh clone
ln -s $O/data/approach1_latent_self_correction/raw data/approach1_latent_self_correction/raw
mkdir -p $E/results $E/q_features logs
cp -r $O/$E/results/correction_v2 $E/results/
ls .venv/bin/python data/approach1_latent_self_correction/raw/docvqa $E/results/correction_v2/train | head
```

The last command should list the Python binary, the DocVQA folder and some `*_shard*.jsonl` files. If it says
"Permission denied", tell Arihanth.

**3. Check the code works with the shared environment** (no GPU needed, a few seconds):

```
.venv/bin/python -m tests.run_tests tests.test_unseen tests.test_correction_scale
bash slurm/check_scripts.sh
```

Both should end without `FAIL`.

**4. Download the 4B model into your copy** (one short CPU job, about 5–10 minutes; it goes to `.hf_cache/` in your copy):

```
sbatch --export=ALL,MODEL_SIZE=4b slurm/prefetch_model.sbatch
squeue -u mjadhav6                     # wait until it is gone
tail -3 logs/prefetch-*.out            # should end without an error
```

**5. Submit the 12 chains** (DocVQA and ScienceQA; train subset, validation and test; two shards each):

```
DATASETS="docvqa scienceqa_img" RANKER=0 bash slurm/run_base_model_pipeline.sh | tee helper_4b_jobs.txt
```

It prints 12 job IDs. Do not set `RANKER=1`: the ranker runs on the main account after all chains are gathered.

**6. Let Arihanth read your outputs.** Your scratch folder is private by default; this lets one user enter it
(the folders inside are already readable):

```
setfacl -m u:ajayavijayan3:--x $(readlink -f ~/scratch)
```

Then send Arihanth the contents of `helper_4b_jobs.txt` and the output of `readlink -f ~/scratch/VLM-LGT`.

## While the jobs run

```
squeue -u mjadhav6                           # state of your jobs
sacct -u mjadhav6 -S today -X -o JobID,JobName%32,State,Elapsed
tail -5 logs/base-model-<job id>.out         # progress of one job
```

- `PENDING (Priority)` or `(QOSGrpGRES)`: waiting for a GPU; nothing to do.
- `PENDING (MaxGRESRunMinsPerUser)`: your GPU-minutes budget is full; the next job starts when one finishes.
- `PENDING (BadConstraints)`: run `scontrol update jobid=<id> Partition=coc-gpu,ice-gpu`.
- **A job ended `TIMEOUT` or `FAILED`:** look at the end of its log. To rerun one shard (it continues where it stopped),
  with the dataset, split and shard from the job name `base-4b-<split>-<dataset>-<shard>`:

  ```
  sbatch -J base-4b-<split>-<dataset>-<shard> -t 10:00:00 \
      --export=ALL,VLMLGT_MODEL=4b,DATASET=<dataset>,SPLIT=<split>,SHARD=<shard>,NUM_SHARDS=2 slurm/run_base_model_chain.sbatch
  ```

- A job is done when its log ends with the `[done]` line of `correction_v3_cot` and `sacct` shows `COMPLETED`.

## What happens with the results

Arihanth copies `results/base_4b/` and `q_features/*_4b.pt` from your copy into the main working copy, then runs the
4B ranker and gating there. You do not need to send files by hand, and nothing in the main copy is written from your
account.

## Not to do

- Do not run anything on the login node except the commands above (no Python experiments outside `sbatch`).
- Do not submit the TextVQA or ChartQA chains: they are done or running on the main account.
- Do not edit or delete anything under `/storage/ice1/9/7/ajayavijayan3/`.
