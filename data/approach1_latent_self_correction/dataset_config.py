# dataset_config.py
#
# Lists the datasets used for Approach 1 (Latent Self-Correction), grouped
# by the role they play: training and grounding signal, the directly
# comparable testbed used in the Self-Correction Mirage study, and harder
# out-of-distribution evaluation sets.
#
# Every entry below was checked against the Hugging Face Hub on
# 2026-09-28, including the ids that turned out to be wrong in an
# earlier version of this file (visual_genome and Ref-L4). Repo ids and
# dataset schemas can still change after that date, so if download.py or
# preprocess.py starts failing for a dataset, check its source_url
# before assuming the code is wrong. See DATASETS.md at the repo root
# for the full reference table and known gotchas.

from data.common.dataset_entry import DatasetEntry

# Training and grounding datasets. These provide the ground truth boxes
# needed to score whether a latent correction moved toward or away from
# the correct region.
TRAINING_DATASETS = {
    "gqa": DatasetEntry(
        hf_repo_id="lmms-lab/GQA",
        source_url="https://huggingface.co/datasets/lmms-lab/GQA",
        notes=(
            "Formatted for the lmms-eval pipeline. Splits are named things "
            "like train_balanced_instructions rather than plain 'train', "
            "and the instruction splits may not carry a bounding box field "
            "the way RefCOCO does. Check the actual schema after download "
            "before assuming normalize_grounding_example applies as-is."
        ),
    ),
    "visual_genome": DatasetEntry(
        hf_repo_id="ranjaykrishna/visual_genome",
        hf_config="region_descriptions_v1.2.0",
        source_url="https://huggingface.co/datasets/ranjaykrishna/visual_genome",
        notes=(
            "The bare 'visual_genome' repo id used in an earlier version "
            "of this file is deprecated and 404s. This dataset has no "
            "default config, region_descriptions_v1.2.0 is the one that "
            "carries bounding boxes, which is what grounding training "
            "needs."
        ),
    ),
    "refcoco": DatasetEntry(
        hf_repo_id="lmms-lab/RefCOCO",
        source_url="https://huggingface.co/datasets/lmms-lab/RefCOCO",
    ),
    "refcocog": DatasetEntry(
        hf_repo_id="lmms-lab/RefCOCOg",
        source_url="https://huggingface.co/datasets/lmms-lab/RefCOCOg",
    ),
}

# Directly comparable testbed. The Self-Correction Mirage study evaluates
# on a specific 2400 train / 510 validation / 505 test mix of RefCOCOg,
# Ref-Adv-S, and Ref-L4. Neither the lmms-lab RefCOCOg split below nor
# the full Ref-L4 release matches that exact split out of the box, so
# reproducing this testbed means subsampling down to the paper's split
# sizes after download, not just concatenating the three datasets as-is.
TESTBED_DATASETS = {
    "refcocog": DatasetEntry(
        hf_repo_id="lmms-lab/RefCOCOg",
        source_url="https://huggingface.co/datasets/lmms-lab/RefCOCOg",
        notes="Only has val/test splits on the Hub, no train split.",
    ),
    "ref_adv_s": DatasetEntry(
        hf_repo_id=None,
        source_url="https://github.com/dddraxxx/Ref-Adv",
        notes=(
            "Not on the Hugging Face Hub. The publicly released subset "
            "(Ref-Adv-s, 1,142 cases with evaluation code and model "
            "predictions, from the ICLR 2026 paper) is distributed from "
            "the paper's GitHub repo. Download it manually and place it "
            "under raw/ref_adv_s before running preprocess.py for the "
            "testbed."
        ),
    ),
    "ref_l4": DatasetEntry(
        hf_repo_id="JierunChen/Ref-L4",
        source_url="https://github.com/JierunChen/Ref-L4",
        notes=(
            "An earlier version of this file listed Ref-L4 as requiring "
            "a manual download; it is actually on the Hub. The full "
            "release has 45,341 referring expressions, subsample it to "
            "match the Mirage study's split sizes rather than using it "
            "whole."
        ),
    ),
}
TESTBED_SPLIT_SIZES = {"train": 2400, "validation": 510, "test": 505}

# Harder and out-of-distribution evaluation sets, covering reasoning-heavy
# cases and small-object or high-resolution stress tests.
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
        notes=(
            "Not gated. License permits commercial use as a test set "
            "only, training on it is not allowed, which fits its role "
            "here as an evaluation-only set."
        ),
    ),
    "vstar_bench": DatasetEntry(
        hf_repo_id="craigwu/vstar_bench",
        source_url="https://huggingface.co/datasets/craigwu/vstar_bench",
        notes="Small by design, 191 examples total in a single test split.",
    ),
    "hr_bench": DatasetEntry(
        hf_repo_id="DreamMr/HR-Bench",
        source_url="https://huggingface.co/datasets/DreamMr/HR-Bench",
        notes=(
            "Ships as TSV files (hr_bench_4k.tsv, hr_bench_8k.tsv) rather "
            "than the usual parquet layout. A plain load_dataset call may "
            "need an explicit config name, check the dataset card if "
            "download.py reports an error for this one."
        ),
    ),
    "vsr": DatasetEntry(
        hf_repo_id="cambridgeltl/vsr_random",
        source_url="https://huggingface.co/datasets/cambridgeltl/vsr_random",
        notes="Splits are named train/dev/test, not train/validation/test.",
    ),
}
