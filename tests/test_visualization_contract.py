"""Lightweight static contracts for the publication visualization layer."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOT_SOURCE = (ROOT / "examples" / "plot_observables.py").read_text(encoding="utf-8")
SUITE_SOURCE = (ROOT / "examples" / "generate_media_suite.py").read_text(encoding="utf-8")
ASYMMETRY_SOURCE = (ROOT / "examples" / "plot_bilateral_asymmetry.py").read_text(encoding="utf-8")


def _constant(name: str):
    tree = ast.parse(PLOT_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing constant: {name}")


def test_public_plot_family_contains_only_the_authorized_auxiliary_addition() -> None:
    family = _constant("COMMON_PLOT_FAMILY")
    assert "auxiliary_kinematics.png" in family
    assert len(family) == 10
    assert "bilateral_force_asymmetry.png" not in family


def test_source_identity_is_not_color_only() -> None:
    assert "def _plot_model" in PLOT_SOURCE
    assert "def _plot_observed" in PLOT_SOURCE
    assert 'marker="o"' in PLOT_SOURCE
    assert "_source_handles" in PLOT_SOURCE
    assert "_side_handles" in PLOT_SOURCE


def test_dense_families_use_structural_redundancy() -> None:
    assert "broken_barh" in PLOT_SOURCE
    assert "plt.subplots(5, 2" in PLOT_SOURCE
    assert "Bar/rack translations" in PLOT_SOURCE
    assert "twinx(" not in PLOT_SOURCE


def test_bilateral_qualification_preserves_side_identity_in_grayscale() -> None:
    assert 'marker="o"' in ASYMMETRY_SOURCE
    assert 'marker="s"' in ASYMMETRY_SOURCE
    assert 'linestyle="--"' in ASYMMETRY_SOURCE


def test_media_manifest_records_visualization_and_plot_hashes() -> None:
    assert '"visualization"' in SUITE_SOURCE
    assert '"plot_sha256"' in SUITE_SOURCE
    assert '"sha256"' in SUITE_SOURCE
