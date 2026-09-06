"""rt_icl -- convert synthetic relational databases into RT's format and pretrain RT with ICL.

Shared by the four per-generator notebooks (RDB-PFN, GRDM, RelDiff, PluRel) so every arm of the
generator study is preprocessed and trained by identical code.
"""

from . import adapters, bench, compat, core, pipeline, prep, subsample, train

__all__ = [
    "adapters", "bench", "compat", "core", "pipeline", "prep", "subsample", "train",
]
__version__ = "0.1.0"
