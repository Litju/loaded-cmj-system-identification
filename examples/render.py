"""Regenerate the full scientific MP4 from the public plant."""

from __future__ import annotations

from pathlib import Path

from loaded_cmj.rendering import main as render_main


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(render_main([
        "--output-dir", str(ROOT / "media" / "20kg_nominal_a"),
        "--params", str(ROOT / "configs" / "synthetic_reference.json"),
        "--output-file", str(ROOT / "media" / "20kg_nominal_a" / "render.mp4"),
        "--trial-id", "20kg_nominal_a",
        "--trial-split", "identification",
    ]))
