"""Regenerate the full scientific MP4 from the public plant."""

from __future__ import annotations

from pathlib import Path

from loaded_cmj.rendering import main as render_main


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(render_main([
        "--output-dir", str(ROOT / "media"),
        "--params", str(ROOT / "configs" / "synthetic_reference.json"),
        "--output-file", str(ROOT / "media" / "loaded_cmj_demo.mp4"),
    ]))

