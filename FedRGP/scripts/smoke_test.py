#!/usr/bin/env python3
"""Run a no-data smoke test for the FedRGP repository."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
METHOD_CONFIG = REPOSITORY_ROOT / "configs/trainers/FedRGP/base2novel_vit_b16.yaml"
DATASETS = {
    "caltech101": "Caltech101",
    "dtd": "DescribableTextures",
    "fgvc_aircraft": "FGVCAircraft",
    "food101": "Food101",
    "oxford_flowers": "OxfordFlowers",
    "oxford_pets": "OxfordPets",
    "ucf101": "UCF101",
}
TEXT_EXTENSIONS = {".md", ".py", ".sh", ".yaml", ".yml"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_repository_files() -> None:
    require(METHOD_CONFIG.is_file(), f"Missing method config: {METHOD_CONFIG}")
    require((REPOSITORY_ROOT / "scripts/FedRGP/base2novel_train.sh").is_file(), "Missing training script")
    require((REPOSITORY_ROOT / "scripts/FedRGP/base2novel_test.sh").is_file(), "Missing test script")

    for dataset_name in DATASETS:
        require(
            (REPOSITORY_ROOT / f"configs/datasets/{dataset_name}.yaml").is_file(),
            f"Missing dataset config: {dataset_name}",
        )
        require(
            (REPOSITORY_ROOT / f"datasets/{dataset_name}.py").is_file(),
            f"Missing dataset loader: {dataset_name}",
        )
        require(
            (REPOSITORY_ROOT / f"train_{dataset_name}.sh").is_file(),
            f"Missing launcher: train_{dataset_name}.sh",
        )


def check_shell_syntax() -> None:
    bash = shutil.which("bash")
    require(bash is not None, "bash is required to validate the launch scripts")

    for script_path in sorted(REPOSITORY_ROOT.rglob("*.sh")):
        subprocess.run([bash, "-n", str(script_path)], check=True)


def check_no_legacy_name() -> None:
    legacy_name = "fed" + "mgp"
    for path in REPOSITORY_ROOT.rglob("*"):
        if legacy_name in path.as_posix().lower():
            raise RuntimeError(f"Legacy method name remains in path: {path}")
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if legacy_name in path.read_text(encoding="utf-8", errors="ignore").lower():
            raise RuntimeError(f"Legacy method name remains in file: {path}")


def check_python_imports_and_configs() -> None:
    os.chdir(REPOSITORY_ROOT)
    sys.path.insert(0, str(REPOSITORY_ROOT))

    from Dassl.dassl.config import get_cfg_default
    from Dassl.dassl.data.datasets import build as dataset_build
    from Dassl.dassl.engine import TRAINER_REGISTRY
    from federated_core.factory import FederatedLearnerFactory
    from federated_main import extend_cfg

    require(FederatedLearnerFactory.get_supported_models() == ["FedRGP"], "FedRGP factory is not registered")
    require(TRAINER_REGISTRY.registered_names() == ["FedRGP"], "FedRGP trainer is not registered")
    require(set(dataset_build._DATASET_IMPORTS) == set(DATASETS.values()), "Dataset registry is inconsistent")

    for dataset_file, dataset_name in DATASETS.items():
        config = get_cfg_default()
        extend_cfg(config)
        config.merge_from_file(str(METHOD_CONFIG))
        config.merge_from_file(str(REPOSITORY_ROOT / f"configs/datasets/{dataset_file}.yaml"))
        require(config.DATASET.NAME == dataset_name, f"Unexpected dataset name for {dataset_file}")
        require(hasattr(config.TRAINER, "FEDRGP"), "Missing TRAINER.FEDRGP configuration")
        dataset_build._maybe_register_dataset(dataset_name)

    require(
        set(dataset_build.DATASET_REGISTRY.registered_names()) == set(DATASETS.values()),
        "Not all dataset loaders can be imported",
    )

    result = subprocess.run(
        [sys.executable, "federated_main.py", "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    require(result.returncode == 0, f"CLI import check failed:\n{result.stderr}")


def main() -> int:
    try:
        check_repository_files()
        check_shell_syntax()
        check_no_legacy_name()
        check_python_imports_and_configs()
    except ModuleNotFoundError as error:
        print(
            f"Missing Python dependency: {error.name}. Run `pip install -r ../requirements.txt` from the source directory.",
            file=sys.stderr,
        )
        return 1
    print("FedRGP smoke test passed: imports, configs, scripts, and naming are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
