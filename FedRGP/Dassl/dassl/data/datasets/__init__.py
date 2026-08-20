"""Dataset registry entrypoints.

Upstream Dassl auto-imports many DA/DG/SSL datasets here. In this FedRGP
codebase, datasets are implemented under the repo-level `datasets/` package,
so we intentionally avoid importing upstream datasets to prevent:
1) name collisions and double-registration errors
2) unnecessary heavy dependencies
"""

from .build import DATASET_REGISTRY, build_dataset  # isort:skip
from .base_dataset import Datum, DatasetBase  # isort:skip
