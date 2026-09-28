# dataset_config.py
#
# Lists the datasets used for Approach 2 (Verifier Reranking). The
# training and out-of-distribution sets match Approach 1, kept here as a
# separate copy so this folder is self-contained and does not depend on
# approach1's config changing underneath it. The one addition specific
# to this approach is the failure-inclusive pool used to train the
# verifier, which needs examples of bad candidates as well as good ones.
#
# Every entry below was checked against the Hugging Face Hub on
# 2026-09-28. See DATASETS.md at the repo root for the full reference
# table and known gotchas (split naming, required configs, file formats).

from data.common.dataset_entry import DatasetEntry

# Same grounding and training sets as Approach 1.
TRAINING_DATASETS = {
    "gqa": DatasetEntry(
        hf_repo_id="lmms-lab/GQA",
        source_url="https://huggingface.co/datasets/lmms-lab/GQA",
        notes="Splits are named like train_balanced_instructions, not plain train.",
    ),
    "visual_genome": DatasetEntry(
        hf_repo_id="ranjaykrishna/visual_genome",
        hf_config="region_descriptions_v1.2.0",
        source_url="https://huggingface.co/datasets/ranjaykrishna/visual_genome",
        notes=(
            "The bare 'visual_genome' repo id is deprecated and 404s. "
            "region_descriptions_v1.2.0 is the config that carries "
            "bounding boxes."
        ),
    ),
    "refcoco": DatasetEntry(
        hf_repo_id="lmms-lab/RefCOCO",
        source_url="https://huggingface.co/datasets/lmms-lab/RefCOCO",
    ),
}

# Same harder and out-of-distribution evaluation sets as Approach 1.
OOD_DATASETS = {
    "chartqa": DatasetEntry(
        hf_repo_id="lmms-lab/ChartQA",
        source_url="https://huggingface.co/datasets/lmms-lab/ChartQA",
    ),
    "ai2d": DatasetEntry(
        hf_repo_id="lmms-lab/ai2d",
        source_url="https://huggingface.co/datasets/lmms-lab/ai2d",
    ),
    "mathvista": DatasetEntry(
        hf_repo_id="AI4Math/MathVista",
        source_url="https://huggingface.co/datasets/AI4Math/MathVista",
        notes="Test-set-only license, do not train on it.",
    ),
    "vstar_bench": DatasetEntry(
        hf_repo_id="craigwu/vstar_bench",
        source_url="https://huggingface.co/datasets/craigwu/vstar_bench",
        notes="Small by design, 191 examples total in a single test split.",
    ),
    "hr_bench": DatasetEntry(
        hf_repo_id="DreamMr/HR-Bench",
        source_url="https://huggingface.co/datasets/DreamMr/HR-Bench",
        notes="Ships as TSV files, may need an explicit config name passed to load_dataset.",
    ),
    "vsr": DatasetEntry(
        hf_repo_id="cambridgeltl/vsr_random",
        source_url="https://huggingface.co/datasets/cambridgeltl/vsr_random",
        notes="Splits are named train/dev/test, not train/validation/test.",
    ),
}

# The verifier needs a training pool that deliberately includes failed
# and uncurated traces, not only successful rollouts. This pool is
# generated locally rather than downloaded, see build_failure_pool.py.
FAILURE_POOL_DIR_NAME = "failure_pool"
