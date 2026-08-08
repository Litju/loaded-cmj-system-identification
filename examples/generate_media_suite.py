"""Generate scenario-centric publication media from the frozen public trials."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import plot_observables as plotting
from loaded_cmj.dataset import load_trial
from loaded_cmj.parameters import load_named_parameters
from loaded_cmj.plant import DT
from loaded_cmj.rendering import DEFAULT_FPS, STEP_STRIDE, WINDOW_START_S, main as render_main
from loaded_cmj.simulation import simulate_trial


ROOT = Path(__file__).resolve().parents[1]
MEDIA_ROOT = ROOT / "media"
PARAMETER_CONFIG = ROOT / "configs" / "synthetic_reference.json"
IDENTIFICATION_DATA = ROOT / "data" / "identification_trials.json"
COMMON_PLOT_FAMILY = plotting.COMMON_PLOT_FAMILY


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _video_info(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = next(stream for stream in payload["streams"] if stream.get("codec_type") == "video")
    fps = float(Fraction(stream["r_frame_rate"]))
    frame_count = stream.get("nb_frames")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frame_count": None if frame_count in (None, "N/A") else int(frame_count),
        "duration_s": float(payload["format"]["duration"]),
        "codec": stream.get("codec_name"),
    }


def _write_preview(render_path: Path, preview_path: Path, takeoff_time_s: float, video_duration_s: float) -> None:
    # The production renderer starts at WINDOW_START_S and plays every STEP_STRIDE
    # physics steps as one video frame.  This selects a real takeoff-near frame.
    preview_time_s = (float(takeoff_time_s) - WINDOW_START_S) / (STEP_STRIDE * DT * DEFAULT_FPS)
    preview_time_s = max(0.25, min(preview_time_s, video_duration_s - 0.25))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{preview_time_s:.6f}",
            "-i", str(render_path), "-frames:v", "1", str(preview_path),
        ],
        check=True,
    )


def _cross_style(ax: Any, title: str, ylabel: str | None = None) -> None:
    ax.set_facecolor(plotting.PANEL)
    ax.set_title(title, loc="left", color=plotting.TEXT, pad=8, fontsize=10, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, color=plotting.TEXT)
    ax.tick_params(axis="both", colors=plotting.MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(plotting.SPINE)
        spine.set_linewidth(0.7)
    ax.grid(True, color=plotting.GRID, alpha=0.52, linewidth=0.55)
    ax.set_axisbelow(True)


def _cross_save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=plotting.BACKGROUND, edgecolor=plotting.BACKGROUND)
    plt.close(fig)


def _event_lines(ax: Any, trial: dict[str, Any], events: dict[str, Any] | None = None) -> None:
    events = trial["observed_events"] if events is None else events
    for key, label in plotting.EVENT_LABELS.items():
        value = events.get(key)
        if value is None:
            continue
        ax.axvline(float(value), color=plotting.EVENT_COLORS[key], linestyle=(0, (3, 2)), linewidth=0.8, alpha=0.85)
        ax.text(float(value), 0.04, label, transform=ax.get_xaxis_transform(), color=plotting.EVENT_COLORS[key],
                fontsize=7, rotation=90, va="bottom", ha="left")


def _plot_scenario_overview(scenarios: list[dict[str, Any]], output: Path, scales: dict[str, tuple[float, float]]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, constrained_layout=True)
    fig.suptitle("Fixed 20 kg public scenarios — force and event overview", color=plotting.TEXT, fontsize=14, fontweight="bold")
    for ax, scenario in zip(axes.flat, scenarios):
        trial = scenario["trial"]
        observations = trial["observations"]
        time_s = np.asarray(observations["time_s"], dtype=float)
        ax.plot(time_s, observations["fz_total_N"], color=plotting.COLORS["total"], linewidth=1.8)
        _event_lines(ax, trial, scenario["result"]["events"])
        _cross_style(ax, scenario["scenario_id"], "total Fz (N)")
        ax.set_ylim(*scales["force_N"])
        ax.set_xlim(float(time_s[0]), float(time_s[-1]))
        ax.text(0.01, 0.90, "observed", transform=ax.transAxes, color=plotting.MUTED, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", color=plotting.TEXT)
    _cross_save(fig, output)


def _plot_grf_comparison(scenarios: list[dict[str, Any]], output: Path, scales: dict[str, tuple[float, float]]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True, constrained_layout=True)
    fig.suptitle("Fixed 20 kg public scenarios — bilateral force-platform comparison", color=plotting.TEXT, fontsize=14, fontweight="bold")
    for ax, scenario in zip(axes.flat, scenarios):
        observations = scenario["trial"]["observations"]
        ax.plot(observations["time_s"], observations["fz_left_N"], color=plotting.COLORS["left"], linewidth=1.55)
        ax.plot(observations["time_s"], observations["fz_right_N"], color=plotting.COLORS["right"], linewidth=1.55, linestyle="--")
        ax.plot(observations["time_s"], observations["fz_total_N"], color=plotting.COLORS["total"], linewidth=2.0)
        _event_lines(ax, scenario["trial"], scenario["result"]["events"])
        _cross_style(ax, scenario["scenario_id"], "force (N)")
        ax.set_ylim(*scales["force_N"])
        ax.text(0.01, 0.90, "Left / Right / Total", transform=ax.transAxes, color=plotting.MUTED, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", color=plotting.TEXT)
    axes[0, 0].legend(handles=[
        Line2D([], [], color=plotting.COLORS["left"], linewidth=2, label="Left"),
        Line2D([], [], color=plotting.COLORS["right"], linewidth=2, linestyle="--", label="Right"),
        Line2D([], [], color=plotting.COLORS["total"], linewidth=2, label="Total"),
    ], loc="upper right", frameon=True, facecolor=plotting.BACKGROUND, edgecolor=plotting.GRID, fontsize=7, labelcolor=plotting.TEXT)
    _cross_save(fig, output)


def _plot_lpt_comparison(scenarios: list[dict[str, Any]], output: Path, scales: dict[str, tuple[float, float]]) -> None:
    fig, axes = plt.subplots(6, 2, figsize=(13, 18), sharex="col", constrained_layout=True)
    fig.suptitle("Fixed 20 kg public scenarios — bar/LPT comparison", color=plotting.TEXT, fontsize=14, fontweight="bold")
    for row, scenario in enumerate(scenarios):
        observations = scenario["trial"]["observations"]
        for column, (key, label, color, group) in enumerate((
            ("bar_displacement_m", "bar displacement (m)", plotting.COLORS["lpt_displacement"], "lpt_displacement_m"),
            ("bar_velocity_m_s", "bar velocity (m/s)", plotting.COLORS["lpt_velocity"], "lpt_velocity_m_s"),
        )):
            ax = axes[row, column]
            ax.plot(observations["time_s"], observations[key], color=color, linewidth=1.75,
                    marker="o", markevery=max(1, len(observations[key]) // 18), markersize=3,
                    markerfacecolor=plotting.BACKGROUND, markeredgecolor=color, markeredgewidth=0.7)
            _event_lines(ax, scenario["trial"], scenario["result"]["events"])
            _cross_style(ax, f"{scenario['scenario_id']} · {label}", label)
            ax.set_ylim(*scales[group])
    axes[-1, 0].set_xlabel("time (s)", color=plotting.TEXT)
    axes[-1, 1].set_xlabel("time (s)", color=plotting.TEXT)
    _cross_save(fig, output)


def _plot_event_timing_comparison(scenarios: list[dict[str, Any]], output: Path) -> None:
    labels = ("onset", "takeoff", "landing")
    keys = ("movement_onset_time_s", "takeoff_time_s", "landing_time_s")
    x = np.arange(len(scenarios), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(1, 1, figsize=(13, 6), constrained_layout=True)
    for offset, (key, label) in enumerate(zip(keys, labels)):
        values = [float(scenario["result"]["events"][key]) for scenario in scenarios]
        ax.bar(x + (offset - 1) * width, values, width, label=label, color=plotting.EVENT_COLORS[key])
    ax.set_xticks(x, [scenario["scenario_id"] for scenario in scenarios], rotation=20, ha="right")
    ax.set_ylabel("event time (s)", color=plotting.TEXT)
    ax.set_xlabel("frozen public scenario", color=plotting.TEXT)
    _cross_style(ax, "Observed event timing across fixed 20 kg scenarios")
    legend = ax.legend(frameon=False, labelcolor=plotting.TEXT)
    for text in legend.get_texts():
        text.set_color(plotting.TEXT)
    _cross_save(fig, output)


def _plot_bilateral_comparison(scenarios: list[dict[str, Any]], output: Path) -> None:
    nominal = next(scenario for scenario in scenarios if scenario["scenario_id"] == "20kg_nominal_a")
    asymmetry = next(scenario for scenario in scenarios if scenario["scenario_id"] == "20kg_bilateral_asymmetry")
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True)
    fig.suptitle("Nominal versus controlled synthetic bilateral drive excitation", color=plotting.TEXT, fontsize=14, fontweight="bold")
    markevery = max(1, len(nominal["trial"]["observations"]["time_s"]) // 18)
    for scenario, linestyle, alpha in ((nominal, "--", 0.68), (asymmetry, "-", 1.0)):
        observations = scenario["trial"]["observations"]
        time_s = observations["time_s"]
        axes[0].plot(time_s, observations["fz_left_N"], color=plotting.COLORS["left"], linestyle=linestyle,
                     alpha=alpha, linewidth=1.8, marker="o", markevery=markevery, markersize=3.0,
                     markerfacecolor=plotting.BACKGROUND, markeredgecolor=plotting.COLORS["left"], markeredgewidth=0.7)
        axes[0].plot(time_s, observations["fz_right_N"], color=plotting.COLORS["right"], linestyle=linestyle,
                     alpha=alpha, linewidth=1.8, marker="s", markevery=markevery, markersize=2.8,
                     markerfacecolor=plotting.BACKGROUND, markeredgecolor=plotting.COLORS["right"], markeredgewidth=0.7)
        difference = np.asarray(observations["fz_left_N"], dtype=float) - np.asarray(observations["fz_right_N"], dtype=float)
        axes[1].plot(time_s, difference, color=plotting.WARM_GOLD, linestyle=linestyle, alpha=alpha, linewidth=1.8)
    _cross_style(axes[0], "Observed bilateral vertical forces", "force (N)")
    _cross_style(axes[1], "Signed physical difference: left Fz − right Fz", "difference (N)")
    axes[1].axhline(0.0, color=plotting.SPINE, linewidth=0.7)
    axes[0].legend(handles=[
        Line2D([], [], color=plotting.COLORS["left"], linewidth=2, marker="o", markersize=4,
               markerfacecolor=plotting.BACKGROUND, label="Left"),
        Line2D([], [], color=plotting.COLORS["right"], linewidth=2, marker="s", markersize=4,
               markerfacecolor=plotting.BACKGROUND, label="Right"),
        Line2D([], [], color=plotting.TEXT, linewidth=2, linestyle="-", label="α = 0.02"),
        Line2D([], [], color=plotting.TEXT, linewidth=1.2, linestyle="--", alpha=0.68, label="Nominal"),
    ], loc="upper left", ncol=2, frameon=True, facecolor=plotting.BACKGROUND, edgecolor=plotting.GRID, fontsize=8, labelcolor=plotting.TEXT)
    axes[1].legend(handles=[Line2D([], [], color=plotting.WARM_GOLD, linewidth=2, label="Left − right"),
                            Line2D([], [], color=plotting.TEXT, linewidth=2, linestyle="-", label="α = 0.02"),
                            Line2D([], [], color=plotting.TEXT, linewidth=1.2, linestyle="--", alpha=0.68, label="Nominal")],
                   loc="upper left", frameon=True, facecolor=plotting.BACKGROUND, edgecolor=plotting.GRID, fontsize=8, labelcolor=plotting.TEXT)
    axes[-1].set_xlabel("time (s)", color=plotting.TEXT)
    _cross_save(fig, output)


def _render_one(
    trial: dict[str, Any],
    result: dict[str, Any],
    output_dir: Path,
    scales: dict[str, tuple[float, float]],
    *,
    render: bool = True,
) -> dict[str, Any]:
    scenario_id = str(trial["trial_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if not result["valid"]:
        raise RuntimeError(f"invalid production rollout for {scenario_id}")
    plot_names = plotting.render_scenario(trial, result, output_dir, scale_context=scales)
    render_path = output_dir / "render.mp4"
    if render:
        render_main([
            "--output-dir", str(output_dir),
            "--params", str(PARAMETER_CONFIG),
            "--output-file", str(render_path),
            "--trial-id", scenario_id,
            "--trial-split", "identification",
        ])
    elif not all(path.exists() for path in (render_path, output_dir / "render_provenance.json")):
        raise FileNotFoundError(f"cannot reuse incomplete render bundle for {scenario_id}")
    video_info = _video_info(render_path)
    preview_path = output_dir / "preview.png"
    if render or not preview_path.exists():
        _write_preview(render_path, preview_path, result["events"]["takeoff_time_s"], video_info["duration_s"])
    from plot_bilateral_asymmetry import main as bilateral_main

    if scenario_id == "20kg_bilateral_asymmetry":
        bilateral_main(["--output", str(output_dir / "bilateral_force_asymmetry.png")])
        plot_names = tuple(plot_names) + ("bilateral_force_asymmetry.png",)
    return {
        "scenario_id": scenario_id,
        "trial": trial,
        "result": result,
        "output_dir": output_dir,
        "plot_names": plot_names,
        "render_path": render_path,
        "preview_path": preview_path,
        "video_info": video_info,
    }


def _manifest_entry(scenario: dict[str, Any], output_root: Path) -> dict[str, Any]:
    trial = scenario["trial"]
    result = scenario["result"]
    output_dir = scenario["output_dir"]
    render_path = scenario["render_path"]
    relative = lambda path: path.relative_to(output_root).as_posix()
    provenance = json.loads((output_dir / "render_provenance.json").read_text())
    video_info = scenario["video_info"]
    trial_hash = _sha256_json(trial)
    return {
        "scenario_id": scenario["scenario_id"],
        "external_load_kg": float(trial["external_load_kg"]),
        "condition": str(trial.get("family_focus", trial.get("trial_group", "public scenario"))),
        "trial_split": "identification",
        "trial_sha256": trial_hash,
        "dataset": "data/identification_trials.json",
        "dataset_sha256": _sha256_file(IDENTIFICATION_DATA),
        "parameter_config": "configs/synthetic_reference.json",
        "parameter_config_sha256": _sha256_file(PARAMETER_CONFIG),
        "render": relative(render_path),
        "preview": relative(scenario["preview_path"]),
        "plots": [relative(output_dir / name) for name in scenario["plot_names"]],
        "render_provenance": relative(output_dir / "render_provenance.json"),
        "plot_sha256": {
            relative(output_dir / name): _sha256_file(output_dir / name)
            for name in scenario["plot_names"]
        },
        "render_resolution": [video_info["width"], video_info["height"]],
        "fps": video_info["fps"],
        "duration_s": video_info["duration_s"],
        "frame_count": video_info["frame_count"] or provenance["rendered_frame_count"],
        "codec": video_info["codec"],
        "generated_from_production_plant": True,
        "plant_model": "assets/loaded_cmj_model.xml",
        "plant_model_sha256": _sha256_file(ROOT / "assets" / "loaded_cmj_model.xml"),
        "makehuman_visual_used": bool(provenance["makehuman_visual_used_in_render"]),
        "qpos_trace_sha256": provenance["qpos_trace_sha256"],
        "qvel_trace_sha256": provenance["qvel_trace_sha256"],
        "observed_events": trial["observed_events"],
        "rollout_events": result["events"],
        "drive_asymmetry_alpha": float(trial.get("drive_asymmetry_alpha", 0.0)),
    }


def generate(output_root: Path = MEDIA_ROOT, *, render: bool = True) -> dict[str, Any]:
    data_manifest = json.loads((ROOT / "data" / "dataset_manifest.json").read_text())
    trial_ids = list(data_manifest["identification_trial_ids"])
    expected = [
        "20kg_nominal_a",
        "20kg_nominal_b",
        "20kg_depth",
        "20kg_timing",
        "20kg_depth_timing",
        "20kg_bilateral_asymmetry",
    ]
    if trial_ids != expected:
        raise RuntimeError(f"final public trial set is not the qualified six: {trial_ids}")
    trials = [load_trial(trial_id, split="identification") for trial_id in trial_ids]
    if any(float(trial["external_load_kg"]) != 20.0 for trial in trials):
        raise RuntimeError("media suite requires every scenario to use external_load_kg=20.0")

    scenarios: list[dict[str, Any]] = []
    for trial in trials:
        scenario_id = trial["trial_id"]
        print(f"Preparing media bundle: {scenario_id}", flush=True)
        result = simulate_trial(load_named_parameters("synthetic_reference"), trial)
        scenarios.append({
            "scenario_id": scenario_id,
            "trial": trial,
            "result": result,
            "output_dir": output_root / scenario_id,
        })

    scales = plotting.build_scale_context(scenarios)
    scenarios = [
        _render_one(
            scenario["trial"], scenario["result"], scenario["output_dir"], scales, render=render,
        )
        for scenario in scenarios
    ]

    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    _plot_scenario_overview(scenarios, comparison_dir / "scenario_overview.png", scales)
    _plot_grf_comparison(scenarios, comparison_dir / "grf_comparison.png", scales)
    _plot_lpt_comparison(scenarios, comparison_dir / "lpt_comparison.png", scales)
    _plot_event_timing_comparison(scenarios, comparison_dir / "event_timing_comparison.png")
    _plot_bilateral_comparison(scenarios, comparison_dir / "bilateral_comparison.png")

    relative = lambda path: path.relative_to(output_root).as_posix()
    manifest = {
        "schema_version": "loaded-cmj-media-v1",
        "scope": "PUBLICATION_SCOPE=20KG_SYSID_V0.1.0",
        "external_load_kg": 20.0,
        "generated_from": {
            "dataset_manifest": "data/dataset_manifest.json",
            "dataset_manifest_sha256": _sha256_file(ROOT / "data" / "dataset_manifest.json"),
            "parameter_config": "configs/synthetic_reference.json",
            "parameter_config_sha256": _sha256_file(PARAMETER_CONFIG),
            "plant_model": "assets/loaded_cmj_model.xml",
            "plant_model_sha256": _sha256_file(ROOT / "assets" / "loaded_cmj_model.xml"),
        },
        "common_plot_family": list(COMMON_PLOT_FAMILY),
        "visualization": {
            "theme": "dark scientific instrument",
            "semantic_palette": "fixed color-blind-safe channel and region palette",
            "source_encoding": "MuJoCo continuous line; observed sparse open markers",
            "side_encoding": "left solid; right dashed where side is not the primary comparison",
            "scenario_scale_policy": "common limits from all six frozen public scenarios",
            "native_units_preserved": True,
        },
        "scenarios": [_manifest_entry(scenario, output_root) for scenario in scenarios],
        "comparison": {
            "scenario_overview": relative(comparison_dir / "scenario_overview.png"),
            "grf_comparison": relative(comparison_dir / "grf_comparison.png"),
            "lpt_comparison": relative(comparison_dir / "lpt_comparison.png"),
            "event_timing_comparison": relative(comparison_dir / "event_timing_comparison.png"),
            "bilateral_comparison": relative(comparison_dir / "bilateral_comparison.png"),
            "sha256": {
                relative(comparison_dir / name): _sha256_file(comparison_dir / name)
                for name in (
                    "scenario_overview.png", "grf_comparison.png", "lpt_comparison.png",
                    "event_timing_comparison.png", "bilateral_comparison.png",
                )
            },
        },
    }
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate all frozen public scenario media bundles.")
    parser.add_argument("--output-root", type=Path, default=MEDIA_ROOT)
    parser.add_argument("--reuse-renders", action="store_true", help="reuse existing verified scenario renders and regenerate plots/manifest")
    args = parser.parse_args(argv)
    manifest = generate(args.output_root.resolve(), render=not args.reuse_renders)
    print(json.dumps({
        "scenario_count": len(manifest["scenarios"]),
        "comparison_count": len(manifest["comparison"]),
        "manifest": str(args.output_root.resolve() / "MANIFEST.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
