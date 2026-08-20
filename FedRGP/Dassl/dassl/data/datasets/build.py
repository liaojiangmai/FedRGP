"""Dataset registry and factory for this repo.

This project vendors Dassl under `Dassl/dassl/` but implements datasets under
the repo-level `datasets/` package.

We register datasets lazily to avoid importing dataset dependencies at module
import time.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

from Dassl.dassl.utils import Registry, check_availability

DATASET_REGISTRY = Registry("DATASET")


# YAML uses these names, e.g. DATASET.NAME: "Caltech101".
_DATASET_IMPORTS: Dict[str, Tuple[str, str]] = {
    "Caltech101": ("datasets.caltech101", "Caltech101"),
    "DescribableTextures": ("datasets.dtd", "DescribableTextures"),
    "FGVCAircraft": ("datasets.fgvc_aircraft", "FGVCAircraft"),
    "Food101": ("datasets.food101", "Food101"),
    "OxfordFlowers": ("datasets.oxford_flowers", "OxfordFlowers"),
    "OxfordPets": ("datasets.oxford_pets", "OxfordPets"),
    "UCF101": ("datasets.ucf101", "UCF101"),
}


def _maybe_register_dataset(name: str) -> None:
    if name in DATASET_REGISTRY.registered_names():
        return
    if name not in _DATASET_IMPORTS:
        return
    module_name, cls_name = _DATASET_IMPORTS[name]
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    DATASET_REGISTRY.register(cls)


def build_dataset(cfg):
    _maybe_register_dataset(cfg.DATASET.NAME)
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)
