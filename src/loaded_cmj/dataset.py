"""Public trial loading and deterministic dataset-generation entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY_ROOT / "data"


def load_dataset(split: str = "identification") -> dict[str, Any]:
    """Load the checked-in identification or validation dataset."""

    filenames = {
        "identification": "identification_trials.json",
        "public": "identification_trials.json",
        "validation": "validation_trials.json",
    }
    try:
        path = DATA_DIR / filenames[split]
    except KeyError as exc:
        raise ValueError(f"unknown dataset split: {split}") from exc
    return json.loads(path.read_text(encoding="utf-8"))


def load_trials(split: str = "identification") -> list[dict[str, Any]]:
    return list(load_dataset(split)["trials"])


def load_trial(trial_id: str, split: str | None = None) -> dict[str, Any]:
    """Load one trial by identifier, optionally constraining its split."""

    splits: Iterable[str] = (split,) if split is not None else ("identification", "validation")
    for active_split in splits:
        for trial in load_trials(active_split):
            if trial.get("trial_id") == trial_id:
                return trial
    raise KeyError(f"trial not found: {trial_id}")


def generate_dataset(*, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Generate reproducible observations using ``tools/generate_dataset.py``.

    The generator is imported lazily so loading a dataset never executes a
    generation side effect. ``output_dir`` can point to a scratch directory for
    regeneration audits; the checked-in fixtures are not modified by default.
    """

    import importlib.util

    generator_path = REPOSITORY_ROOT / "tools" / "generate_dataset.py"
    spec = importlib.util.spec_from_file_location("loaded_cmj_dataset_generator", generator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load dataset generator from {generator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate(output_dir=output_dir)

