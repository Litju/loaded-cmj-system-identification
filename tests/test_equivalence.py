from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

SOURCE_ROOT = os.environ.get("LOADED_CMJ_SOURCE_ROOT")


@pytest.mark.skipif(SOURCE_ROOT is None, reason="set LOADED_CMJ_SOURCE_ROOT for the read-only source audit")
def test_source_target_equivalence() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.equivalence_harness import run_equivalence

    report = run_equivalence(Path(SOURCE_ROOT), trial_id="20kg_nominal_a", tolerance=1e-12)
    assert report["compiled_model_equivalence"] == "PASS"
    assert report["trajectory_equivalence"] == "PASS"
    assert report["measurement_equivalence"] == "PASS"
    assert report["event_equivalence"] == "PASS"
