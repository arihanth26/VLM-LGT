# dataset_entry.py
#
# Shared data structure describing one dataset entry, used by both
# approach1 and approach2's dataset_config.py files. Centralizing this
# here means both approaches describe their datasets the same way, and a
# download or preprocess script never has to guess whether a config
# value is a plain string or something richer.

from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetEntry:
    """Describes where one dataset comes from and how to load it.

    hf_repo_id: Hugging Face Hub repo id, or None if the dataset is not
        on the Hub and needs to be sourced manually.
    hf_config: the dataset config or subset name to pass to
        datasets.load_dataset, for datasets that require one (Visual
        Genome, for example, has no default config). Leave as None for
        datasets that load fine without specifying one.
    source_url: a GitHub repo, paper, or dataset card URL, kept here so
        anyone using this dataset can check the original source instead
        of trusting the id alone.
    notes: anything a teammate should know before using this dataset,
        such as a schema quirk, a split naming difference, or a
        licensing restriction.
    """

    hf_repo_id: Optional[str]
    hf_config: Optional[str] = None
    source_url: Optional[str] = None
    notes: str = ""
