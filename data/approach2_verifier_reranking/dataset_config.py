# dataset_config.py
#
# Lists the datasets used for Approach 2 (Verifier Reranking). The
# training and out-of-distribution sets match Approach 1, kept here as a
# separate copy so this folder is self-contained and does not depend on
# approach1's config changing underneath it. The one addition specific
# to this approach is the failure-inclusive pool used to train the
# verifier, which needs examples of bad candidates as well as good ones.

# Same grounding and training sets as Approach 1.
TRAINING_DATASETS = {
    "gqa": "lmms-lab/GQA",
    "visual_genome": "visual_genome",
    "refcoco": "lmms-lab/RefCOCO",
}

# Same harder and out-of-distribution evaluation sets as Approach 1.
OOD_DATASETS = {
    "chartqa": "lmms-lab/ChartQA",
    "ai2d": "lmms-lab/ai2d",
    "mathvista": "AI4Math/MathVista",
    "vstar_bench": "craigwu/vstar_bench",
    "hr_bench": "DreamMr/HR-Bench",
    "vsr": "cambridgeltl/vsr_random",
}

# The verifier needs a training pool that deliberately includes failed
# and uncurated traces, not only successful rollouts. This pool is
# generated locally rather than downloaded, see build_failure_pool.py.
FAILURE_POOL_DIR_NAME = "failure_pool"
