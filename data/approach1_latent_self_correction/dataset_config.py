# dataset_config.py
#
# Lists the datasets used for Approach 1 (Latent Self-Correction), grouped
# by the role they play: training and grounding signal, the directly
# comparable testbed used in the Self-Correction Mirage study, and harder
# out-of-distribution evaluation sets. Hugging Face dataset ids are given
# where the dataset is available on the Hub; a few sets need a manual
# download step, which is noted next to them below.

# Training and grounding datasets. These provide the ground truth boxes
# needed to score whether a latent correction moved toward or away from
# the correct region.
TRAINING_DATASETS = {
    "gqa": "lmms-lab/GQA",
    "visual_genome": "visual_genome",
    "refcoco": "lmms-lab/RefCOCO",
    "refcocog": "lmms-lab/RefCOCOg",
}

# Directly comparable testbed. Matches the three-dataset mix and split
# (2400 train, 510 validation, 505 test) used in the Self-Correction
# Mirage study, so results here can be compared against a documented
# result on identical data. Ref-Adv-S and Ref-L4 are not on the Hugging
# Face Hub at the time of writing and need to be requested separately;
# download.py checks for a local copy and reports if one is missing
# instead of failing silently.
TESTBED_DATASETS = {
    "refcocog": "lmms-lab/RefCOCOg",
    "ref_adv_s": None,  # manual download required, see download.py
    "ref_l4": None,  # manual download required, see download.py
}
TESTBED_SPLIT_SIZES = {"train": 2400, "validation": 510, "test": 505}

# Harder and out-of-distribution evaluation sets, covering reasoning-heavy
# cases and small-object or high-resolution stress tests.
OOD_DATASETS = {
    "chartqa": "lmms-lab/ChartQA",
    "ai2d": "lmms-lab/ai2d",
    "mathvista": "AI4Math/MathVista",
    "vstar_bench": "craigwu/vstar_bench",
    "hr_bench": "DreamMr/HR-Bench",
    "vsr": "cambridgeltl/vsr_random",
}
