#!/usr/bin/env python3
"""Truthful scientific render of a loaded-CMJ rollout.

Every rendered pixel corresponds to an exact MuJoCo state produced by the
authoritative public rollout ``plant.run_trial`` (deterministic ``mujoco.mj_step``
integration of the committed morphology ``assets/loaded_cmj_model.xml``, which the
plant loads with ``mujoco.MjModel.from_xml_path``). The renderer never advances
its own physics and never writes ``data.qpos`` / ``data.qvel``: it installs a
read-only capture-and-draw hook around ``mujoco.mj_step`` so the frames are the
live post-step states of the real rollout, records those states for hashing, and
draws honest schematic overlays anchored to actual MuJoCo body/site positions.

Design contract (5E-D-R3):
  * The committed morphology is the sole dynamics plant. A separate MakeHuman
    model is loaded only by this renderer as a visual overlay; it is never
    stepped and never used by plant physics or data generation.
  * No fake anatomy: visual overlays are driven from exact live CMJ body/site
    states; plant state, contact, COM, CoP, GRF, and LPT signals are read-only.
  * Fixed sagittal world camera anchored to the force plate so the ~0.31 m jump
    is readable against a fixed ground line and vertical ruler.
  * The LPT tether is shown as a thin measurement line only (zero-force here).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import mujoco
import numpy as np


DEFAULT_FPS = 30
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
# Playback pacing: one rendered frame per STEP_STRIDE physics steps. With dt=2 ms
# and 30 fps this yields a labelled slow-motion factor of STEP_STRIDE*dt*fps.
STEP_STRIDE = 4
WINDOW_START_S = 1.00  # begin the shown window inside the quiet-stance phase
END_HOLD_FRAMES = 10   # short readable hold on the settled final pose

# Fixed sagittal camera (world-anchored, not COM-tracking). Raised/backed off from
# the pre-Slice-A framing so the taller whole-humanoid upper body (head ~1.84 m),
# the loaded bar, the front LPT device, and the airborne apex are all in frame.
CAM_LOOKAT = (0.12, 0.0, 0.98)
CAM_DISTANCE = 3.55
CAM_AZIMUTH = 102.0
CAM_ELEVATION = -6.0
# COM marker/path is anchored to the exact MuJoCo COM. A post-render 2D
# foreground overlay projects the same exact COM x/z into screen space so the
# reviewer marker remains visible even when the body/bar depth-occlude it.
COM_FRONT_OFFSET_M = 0.0
COM_SCREEN_TRAIL_STEPS = 220

# Joint / landmark trace layer. Every entry maps to an EXACT MuJoCo site in the
# Slice-A model; positions are read (never authored) from data.site_xpos. COM comes
# from data.subtree_com and the bar/LPT attach from bar_lpt_site (see
# SLICE_B_TRACE_SITE_MAP). Joints render a short fading recent trail + marker; COM
# and bar/LPT render the full path so the two measurement channels read clearly.
JOINT_TRACE_SITES = (
    ("root_site", "PELVIS"),
    ("left_hip_site", "HIP"), ("right_hip_site", "HIP"),
    ("left_knee_site", "KNEE"), ("right_knee_site", "KNEE"),
    ("left_ankle_site", "ANKLE"), ("right_ankle_site", "ANKLE"),
)
FOOT_TRACE_SITES = (
    "left_heel_site", "left_forefoot_site", "left_toe_site",
    "right_heel_site", "right_forefoot_site", "right_toe_site",
)
COL_JOINT = (0.98, 0.74, 0.30, 1.0)   # amber joints
COL_BAR_TRACE = (0.35, 0.85, 0.98, 1.0)  # cyan bar/LPT attach, distinct from COM yellow
COL_LPT_DEVICE = (0.30, 0.80, 0.92, 1.0)
JOINT_TRAIL_STEPS = 130  # ~0.26 s of fading recent history at dt=2 ms

# --------------------------------------------------------------------------- #
# MakeHuman render-only visual subject. The committed morphology is UNCHANGED.
# The final visible athlete is the repository-local segmented MakeHuman mesh, retargeted
# to exact plant skeleton sites and depth-composited over the truthful instrumented
# scene. Plant body capsules are transparent in render so there is not a second
# overlapping athlete.
MAKEHUMAN_VISUAL_RELATIVE_XML = Path("assets/makehuman_cmj_visual/makehuman_segment_visual.xml")
MAKEHUMAN_OBJ_RELATIVE_PATH = Path("assets/makehuman_cmj_visual/makehuman_cmj_body.obj")
MAKEHUMAN_DAE_RELATIVE_PATH = Path("assets/makehuman_cmj_visual/makehuman_cmj_body.dae")
MAKEHUMAN_MANIFEST_RELATIVE_PATH = Path("assets/makehuman_cmj_visual/segments/segment_manifest.json")
# True MakeHuman LBS skin (MuJoCo-native <skin>), generated offline from the
# vendored DAE by assets/makehuman_cmj_visual/skin/dae_to_mujoco_skin_converter.py.
# This is the PRIMARY human render: one continuous soft-tissue surface driven by
# 163 render-only mocap "bone" bodies retargeted from exact live CMJ landmarks.
MAKEHUMAN_SKIN_RELATIVE_XML = Path("assets/makehuman_cmj_visual/skin/makehuman_skin_visual.xml")
MAKEHUMAN_SKIN_BONES_RELATIVE_JSON = Path("assets/makehuman_cmj_visual/skin/makehuman_skin_bones.json")
MAKEHUMAN_SKIN_RGBA = (0.79, 0.58, 0.45, 1.0)
BAR_STEEL_RGBA = (0.55, 0.58, 0.63, 1.0)
BAR_PLATE_RGBA = (0.08, 0.09, 0.11, 1.0)
BAR_COLLAR_RGBA = (0.24, 0.26, 0.30, 1.0)
LPT_BODY_RGBA = (0.055, 0.065, 0.075, 1.0)   # compact black LPT housing
LPT_ACCENT_RGBA = (0.16, 0.82, 0.90, 1.0)    # subtle teal/cyan accent lines

# The plant's own humanoid/contact geoms are transparent for rendering only
# (physics untouched). A single rigged scene skin and explicit barbell are drawn
# from exact plant sites/body transforms instead.
GEOM_HIDE_HUMANOID = (
    "head_geom", "neck_geom", "thorax_geom", "abdomen_geom", "pelvis_geom", "sacrum_geom",
    "left_upper_arm_geom", "right_upper_arm_geom", "left_forearm_geom", "right_forearm_geom",
    "left_thigh_geom", "right_thigh_geom", "left_shank_geom", "right_shank_geom",
    "left_hindfoot_geom", "right_hindfoot_geom",
    "left_forefoot_geom_struct", "right_forefoot_geom_struct",
    "left_toe_geom_struct", "right_toe_geom_struct",
    "left_heel", "left_forefoot", "left_toe", "right_heel", "right_forefoot", "right_toe",
)
# Force-plate visual deck at HALF the previous rendered footprint. The physical
# plate geom (half-extents 1.25 x 0.50) is unchanged and omitted from the visual;
# this clean lab deck is what the reviewer sees, still under both feet.
FORCE_PLATE_VISUAL_HALF = (0.62, 0.26)   # x, y half-extents (was ~1.25 x 0.50)

MAKEHUMAN_SEGMENT_LOCAL_AXES = {
    "mh_head": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "mh_torso": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "mh_pelvis": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "mh_left_thigh": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    "mh_right_thigh": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    "mh_left_shank": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    "mh_right_shank": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    "mh_left_foot": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "mh_right_foot": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "mh_left_upper_arm": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "mh_right_upper_arm": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "mh_left_forearm": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "mh_right_forearm": np.array([0.0, -1.0, 0.0], dtype=np.float64),
}
VISUAL_ARM_UPPER_LEN_M = 0.285
VISUAL_ARM_FORE_LEN_M = 0.290
# The plant ankle->toe landmark span (~0.23 m) is much longer than the MakeHuman
# bind foot (~0.14 m); a 1:1 retarget stretches the heavily-modeled toe mesh into
# an overlong "flipper". These compact the visual foot toward natural proportion
# (foot length / body height ~0.16) while keeping the sole seated on the deck.
FOOT_VISUAL_LEN_SCALE = 0.70
FOOT_VISUAL_SOLE_DROP_M = 0.006
JOINT_LANDMARK_RED = (255, 35, 35)
JOINT_LANDMARK_OUTLINE = (8, 10, 14)

# Anatomical landmark layer: RED = exact CMJ joint-center sites that register under the
# visible MakeHuman skin (verified inside the skin silhouette every phase), mirrored by
# the JOINT KINEMATICS graph rows. The MTP joints are intentionally NOT in this red layer:
# the visual foot is compacted (FOOT_VISUAL_LEN_SCALE) so the exact MTP site projects
# 1-11 px beyond the visible forefoot, i.e. it is not registered to visible foot anatomy.
# MTP kinematics are still shown in the JOINT KINEMATICS panel ("MTP L/R"), and the foot
# ground interface is shown by the separate green/cyan heel/forefoot/toe CONTACT markers.
LANDMARK_SITES = (
    ("pelvis_site", "PELVIS"),
    ("lumbar_site", "LUMBAR"),
    ("left_hip_site", "HIP"), ("right_hip_site", "HIP"),
    ("left_knee_site", "KNEE"), ("right_knee_site", "KNEE"),
    ("left_ankle_site", "ANKLE"), ("right_ankle_site", "ANKLE"),
)
# Sites that carry a fading screen-space trace (the primary kinematic joints).
LANDMARK_TRACE_SITES = (
    "left_hip_site", "right_hip_site", "left_knee_site", "right_knee_site",
    "left_ankle_site", "right_ankle_site",
)

# Overlay palette (RGB 0-255 for the 2D panel, 0-1 rgba for scene geoms).
COL_BG_PANEL = (10, 13, 18)
COL_TEXT = (232, 236, 242)
COL_DIM = (150, 158, 170)
COL_ACCENT = (94, 196, 235)
COL_COM = (247, 208, 72)
COL_FORCE = (120, 210, 150)
COL_EVENT = (250, 130, 96)
COL_BAR_TRACE_255 = (90, 216, 250)  # cyan, matches the 3D bar/LPT trace
LPT_SCENE_LABEL = "LPT"
LPT_TETHER_TELEMETRY_LABEL = "LPT tether = measurement-only"
LEFT_JOINT_PANEL_X0 = 16
LEFT_JOINT_PANEL_X1 = 440
LEFT_JOINT_PANEL_Y0 = 54
LEFT_JOINT_PANEL_BOTTOM_MARGIN_PX = 210
LOWER_LEFT_TELEMETRY_BOUNDS = (16, 178, 392, 16)  # x0, height-from-bottom, x1, bottom margin
RIGHT_DASHBOARD_WIDTH_PX = 507
RIGHT_METER_WIDTH_PX = 32
RIGHT_METER_LABEL_GUTTER_PX = 60
RIGHT_METER_RIGHT_LABEL_GUTTER_PX = 32
RIGHT_TIME_PLOT_LEFT_INSET_PX = 0
RIGHT_GRF_PANEL_Y0 = 54
RIGHT_GRF_PANEL_Y1 = 222
RIGHT_FZ_GRAPH_Y0 = 300
RIGHT_FZ_GRAPH_BOTTOM_MARGIN_PX = 116
PHASE_COLORS_255 = {
    "WEIGHING": (120, 128, 140),
    "UNWEIGHTING": (86, 150, 214),
    "BRAKING": (232, 160, 74),
    "PROPULSION": (226, 96, 88),
    "FLIGHT": (108, 200, 128),
    "LANDING_ABSORPTION": (170, 120, 210),
    "STABILIZATION": (96, 190, 190),
}
PHASE_SHORT = {
    "WEIGHING": "WEIGHING",
    "UNWEIGHTING": "UNWEIGHTING",
    "BRAKING": "BRAKING",
    "PROPULSION": "PROPULSION",
    "FLIGHT": "FLIGHT",
    "LANDING_ABSORPTION": "LANDING",
    "STABILIZATION": "STABILIZATION",
}
PHASE_LABELS = {
    "WEIGHING": "WEIGHING",
    "UNWEIGHTING": "UNWEIGHTING",
    "BRAKING": "BRAKING",
    "PROPULSION": "PROPULSION",
    "FLIGHT": "FLIGHT",
    "LANDING_ABSORPTION": "LANDING",
    "STABILIZATION": "STABILIZATION",
}
PHASE_ABBR = {
    "WEIGHING": "WGH",
    "UNWEIGHTING": "UNW",
    "BRAKING": "BRK",
    "PROPULSION": "PROP",
    "FLIGHT": "FLT",
    "LANDING_ABSORPTION": "LAND",
    "STABILIZATION": "STAB",
}

REQUESTED_JOINT_AUDIT = (
    "root_x",
    "root_z",
    "root_pitch",
    "lumbar_pitch",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_forefoot_rocker",
    "right_forefoot_rocker",
    "left_mtp",
    "right_mtp",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "bar_rack_x",
    "bar_rack_z",
    "bar_rack_pitch",
)

JOINT_PANEL_ITEMS = (
    ("Pelvis Z", ("root_z",)),
    ("Trunk Pitch", ("root_pitch",)),
    ("Lumbar Angle", ("lumbar_pitch",)),
    ("Hip Angle L/R", ("left_hip", "right_hip")),
    ("Knee Angle L/R", ("left_knee", "right_knee")),
    ("Ankle Angle L/R", ("left_ankle", "right_ankle")),
    ("Forefoot Rocker L/R", ("left_forefoot_rocker", "right_forefoot_rocker")),
    ("Toe / MTP L/R", ("left_mtp", "right_mtp")),
    ("Bar Height", ("bar_rack_z",)),
)


# --------------------------------------------------------------------------- #
# Minimal dependency-free 5x7 bitmap font + numpy drawing primitives.         #
# --------------------------------------------------------------------------- #
_FONT_ROWS = {
    " ": ("     ", "     ", "     ", "     ", "     ", "     ", "     "),
    "0": (" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "),
    "1": ("  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "),
    "2": (" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"),
    "3": (" ### ", "#   #", "    #", "  ## ", "    #", "#   #", " ### "),
    "4": ("   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "),
    "5": ("#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "),
    "6": (" ### ", "#   #", "#    ", "#### ", "#   #", "#   #", " ### "),
    "7": ("#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "),
    "8": (" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "),
    "9": (" ### ", "#   #", "#   #", " ####", "    #", "#   #", " ### "),
    "A": (" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    "B": ("#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "),
    "C": (" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "),
    "D": ("###  ", "#  # ", "#   #", "#   #", "#   #", "#  # ", "###  "),
    "E": ("#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"),
    "F": ("#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "),
    "G": (" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ### "),
    "H": ("#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    "I": (" ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "),
    "J": ("  ###", "   # ", "   # ", "   # ", "#  # ", "#  # ", " ##  "),
    "K": ("#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"),
    "L": ("#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"),
    "M": ("#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #"),
    "N": ("#   #", "##  #", "##  #", "# # #", "#  ##", "#  ##", "#   #"),
    "O": (" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "),
    "P": ("#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "),
    "Q": (" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"),
    "R": ("#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"),
    "S": (" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "),
    "T": ("#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "),
    "U": ("#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "),
    "V": ("#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "),
    "W": ("#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"),
    "X": ("#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"),
    "Y": ("#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "),
    "Z": ("#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"),
    ".": ("     ", "     ", "     ", "     ", "     ", " ##  ", " ##  "),
    ",": ("     ", "     ", "     ", "     ", " ##  ", "  #  ", " #   "),
    ":": ("     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", "     "),
    "-": ("     ", "     ", "     ", "#####", "     ", "     ", "     "),
    "+": ("     ", "  #  ", "  #  ", "#####", "  #  ", "  #  ", "     "),
    "/": ("    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    "),
    "(": ("  ## ", " #   ", "#    ", "#    ", "#    ", " #   ", "  ## "),
    ")": (" ##  ", "   # ", "    #", "    #", "    #", "   # ", " ##  "),
    "=": ("     ", "     ", "#####", "     ", "#####", "     ", "     "),
    "%": ("##  #", "##  #", "   # ", "  #  ", " #   ", "#  ##", "#  ##"),
    "<": ("   # ", "  #  ", " #   ", "#    ", " #   ", "  #  ", "   # "),
    ">": (" #   ", "  #  ", "   # ", "    #", "   # ", "  #  ", " #   "),
}
_GLYPH_CACHE: dict[str, np.ndarray] = {}


def _glyph(ch: str) -> np.ndarray:
    key = ch if ch in _FONT_ROWS else "?"
    if key == "?" and "?" not in _FONT_ROWS:
        key = " "
    cached = _GLYPH_CACHE.get(key)
    if cached is None:
        rows = _FONT_ROWS[key]
        cached = np.array([[1 if c == "#" else 0 for c in row] for row in rows], dtype=np.uint8)
        _GLYPH_CACHE[key] = cached
    return cached


def _text_width(text: str, scale: int, spacing: int) -> int:
    if not text:
        return 0
    return len(text) * (5 * scale + spacing) - spacing


def draw_text(img: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int],
              scale: int = 2, spacing: int = 1) -> int:
    h, w = img.shape[:2]
    col = np.array(color, dtype=np.uint8)
    cx = x
    for ch in text.upper():
        glyph = _glyph(ch)
        block = np.repeat(np.repeat(glyph, scale, axis=0), scale, axis=1)
        gh, gw = block.shape
        x0 = cx
        y0 = y
        x1 = min(w, x0 + gw)
        y1 = min(h, y0 + gh)
        if x0 < w and y0 < h and x1 > 0 and y1 > 0:
            sub = block[max(0, -y0):y1 - y0, max(0, -x0):x1 - x0]
            region = img[max(0, y0):y1, max(0, x0):x1]
            mask = sub > 0
            region[mask] = col
        cx += gw + spacing
    return cx


def fill_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
              color: tuple[int, int, int], alpha: float = 1.0) -> None:
    h, w = img.shape[:2]
    x0b = max(0, int(x0)); y0b = max(0, int(y0))
    x1b = min(w, int(x1)); y1b = min(h, int(y1))
    if x1b <= x0b or y1b <= y0b:
        return
    region = img[y0b:y1b, x0b:x1b].astype(np.float32)
    col = np.array(color, dtype=np.float32)
    a = float(max(0.0, min(1.0, alpha)))
    blended = region * (1.0 - a) + col * a
    img[y0b:y1b, x0b:x1b] = blended.astype(np.uint8)


def rect_outline(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                 color: tuple[int, int, int], t: int = 1) -> None:
    fill_rect(img, x0, y0, x1, y0 + t, color)
    fill_rect(img, x0, y1 - t, x1, y1, color)
    fill_rect(img, x0, y0, x0 + t, y1, color)
    fill_rect(img, x1 - t, y0, x1, y1, color)


def draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
              color: tuple[int, int, int], t: int = 1, alpha: float = 1.0) -> None:
    steps = max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0)), 1)
    r = max(0, int(t) // 2)
    for k in range(steps + 1):
        u = k / steps
        x = int(round(x0 + (x1 - x0) * u))
        y = int(round(y0 + (y1 - y0) * u))
        fill_rect(img, x - r, y - r, x + r + 1, y + r + 1, color, alpha)


def draw_polyline(img: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int],
                  t: int = 1, alpha: float = 1.0) -> None:
    for a, b in zip(points[:-1], points[1:]):
        draw_line(img, a[0], a[1], b[0], b[1], color, t=t, alpha=alpha)


def draw_circle(img: np.ndarray, cx: int, cy: int, radius: int,
                color: tuple[int, int, int], alpha: float = 1.0) -> None:
    h, w = img.shape[:2]
    r = max(1, int(radius))
    x0 = max(0, int(cx) - r)
    x1 = min(w, int(cx) + r + 1)
    y0 = max(0, int(cy) - r)
    y1 = min(h, int(cy) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - int(cx)) ** 2 + (yy - int(cy)) ** 2 <= r * r
    region = img[y0:y1, x0:x1].astype(np.float32)
    col = np.array(color, dtype=np.float32)
    a = float(max(0.0, min(1.0, alpha)))
    region[mask] = region[mask] * (1.0 - a) + col * a
    img[y0:y1, x0:x1] = region.astype(np.uint8)


def _project_world_to_screen(scene: Any, pos: np.ndarray, width: int,
                             height: int) -> tuple[int, int] | None:
    """Project a MuJoCo world point with the live render camera.

    This is render-only instrumentation: it does not move the COM in world space.
    The projected point is used after ``renderer.render()`` for a foreground 2D
    overlay that cannot be depth-occluded by the humanoid or bar.
    """
    try:
        cam = scene.camera[0]
        cam_pos = np.asarray(cam.pos, dtype=np.float64)
        forward = np.asarray(cam.forward, dtype=np.float64)
        up = np.asarray(cam.up, dtype=np.float64)
        near = float(cam.frustum_near)
        frustum_width = float(cam.frustum_width)
        frustum_center = float(cam.frustum_center)
        frustum_bottom = float(cam.frustum_bottom)
        frustum_top = float(cam.frustum_top)
        orthographic = bool(cam.orthographic)
    except Exception:
        return None
    if frustum_width <= 0.0:
        frustum_width = max(frustum_top - frustum_bottom, 1e-9) * (float(width) / float(height))

    forward_norm = np.linalg.norm(forward)
    up_norm = np.linalg.norm(up)
    if forward_norm <= 0.0 or up_norm <= 0.0:
        return None
    forward = forward / forward_norm
    up = up / up_norm
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm <= 0.0:
        return None
    right = right / right_norm

    rel = np.asarray(pos, dtype=np.float64) - cam_pos
    z = float(np.dot(rel, forward))
    x = float(np.dot(rel, right))
    y = float(np.dot(rel, up))
    left = frustum_center - 0.5 * frustum_width
    right_plane = frustum_center + 0.5 * frustum_width
    if orthographic:
        px_plane = x
        py_plane = y
    else:
        if z <= max(near, 1e-6):
            return None
        scale = near / z
        px_plane = x * scale
        py_plane = y * scale
    if right_plane <= left or frustum_top <= frustum_bottom:
        return None
    u = (px_plane - left) / (right_plane - left)
    v = (py_plane - frustum_bottom) / (frustum_top - frustum_bottom)
    return int(round(u * width)), int(round((1.0 - v) * height))


# --------------------------------------------------------------------------- #
# Plant loading                                                               #
# --------------------------------------------------------------------------- #
def _load_public_plant(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("loaded_cmj_public_plant", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load plant module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bound(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


# --------------------------------------------------------------------------- #
# Scene overlay geoms (visual-only, anchored to real MuJoCo positions)        #
# --------------------------------------------------------------------------- #
def _add_sphere(scene: Any, pos, radius: float, rgba, emission: float = 0.35) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.array(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(9),
        np.array(rgba, dtype=np.float32),
    )
    g.emission = float(emission)
    scene.ngeom += 1


class _MakeHumanSegmentRenderer:
    """Render-only MakeHuman soft-tissue visual subject.

    The MakeHuman OBJ is split into opaque body-surface segments and
    loaded in a separate MuJoCo model. Each segment has a free joint in that
    separate visual model; every frame those joints are set from exact live CMJ
    sites/body poses and then mj_forward is called for rendering only.
    """

    def __init__(self, task_dir: Path, width: int, height: int, cam_lookat,
                 cam_distance: float, cam_azimuth: float, cam_elevation: float) -> None:
        self.asset_xml = (task_dir / MAKEHUMAN_VISUAL_RELATIVE_XML).resolve()
        self.asset_obj = (task_dir / MAKEHUMAN_OBJ_RELATIVE_PATH).resolve()
        self.asset_dae = (task_dir / MAKEHUMAN_DAE_RELATIVE_PATH).resolve()
        self.manifest_path = (task_dir / MAKEHUMAN_MANIFEST_RELATIVE_PATH).resolve()
        self.width = int(width)
        self.height = int(height)
        self.available = False
        self.load_error: str | None = None
        self.frames_rendered = 0
        self.model: Any = None
        self.data: Any = None
        self.renderer: Any = None
        self.camera: Any = None
        self.option: Any = None
        self.jadr: dict[str, int] = {}
        self.cmj_sid: dict[str, int] = {}
        self.cmj_bid: dict[str, int] = {}
        self.segment_manifest: dict[str, Any] = {}
        self.last_left_hand: np.ndarray | None = None
        self.last_right_hand: np.ndarray | None = None

        missing = [str(p) for p in (self.asset_xml, self.asset_obj, self.manifest_path) if not p.exists()]
        if missing:
            self.load_error = "missing MakeHuman visual asset(s): " + ", ".join(missing)
            return
        try:
            self.segment_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.model = mujoco.MjModel.from_xml_path(str(self.asset_xml))
            self.data = mujoco.MjData(self.model)
            self.model.vis.global_.offwidth = self.width
            self.model.vis.global_.offheight = self.height
            self.model.vis.headlight.ambient[:] = [0.56, 0.56, 0.56]
            self.model.vis.headlight.diffuse[:] = [0.62, 0.62, 0.62]
            self.model.vis.headlight.specular[:] = [0.08, 0.08, 0.08]
            for name in MAKEHUMAN_SEGMENT_LOCAL_AXES:
                jid = mujoco.mj_name2id(self.model, int(mujoco.mjtObj.mjOBJ_JOINT), f"{name}_free")
                if jid < 0:
                    raise RuntimeError(f"MakeHuman segment joint not found: {name}_free")
                self.jadr[name] = int(self.model.jnt_qposadr[jid])
            skin = np.array(MAKEHUMAN_SKIN_RGBA, dtype=np.float32)
            for gid in range(self.model.ngeom):
                rgba = skin.copy()
                body_id = int(self.model.geom_bodyid[gid])
                body_name = mujoco.mj_id2name(self.model, int(mujoco.mjtObj.mjOBJ_BODY), body_id)
                if body_name in (
                    "mh_left_upper_arm", "mh_left_forearm",
                    "mh_right_upper_arm", "mh_right_forearm",
                    "mh_left_foot", "mh_right_foot",
                ):
                    rgba[3] = 0.0
                self.model.geom_rgba[gid] = rgba
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = list(cam_lookat)
            cam.distance = float(cam_distance)
            cam.azimuth = float(cam_azimuth)
            cam.elevation = float(cam_elevation)
            self.camera = cam
            opt = mujoco.MjvOption()
            opt.sitegroup[:] = 0
            opt.flags[int(mujoco.mjtVisFlag.mjVIS_TENDON)] = 0
            self.option = opt
            self.available = True
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.available = False

    def bind_cmj(self, cmj_model: Any) -> None:
        for sname in (
            "pelvis_site", "sternum_site", "head_site",
            "left_hip_site", "right_hip_site", "left_knee_site", "right_knee_site",
            "left_ankle_site", "right_ankle_site", "left_heel_site", "right_heel_site",
            "left_toe_site", "right_toe_site", "left_bar_grip_site", "right_bar_grip_site",
        ):
            sid = mujoco.mj_name2id(cmj_model, int(mujoco.mjtObj.mjOBJ_SITE), sname)
            if sid < 0:
                raise RuntimeError(f"CMJ site not found for MakeHuman visual: {sname}")
            self.cmj_sid[sname] = int(sid)
        for bname in ("left_upper_arm", "right_upper_arm"):
            bid = mujoco.mj_name2id(cmj_model, int(mujoco.mjtObj.mjOBJ_BODY), bname)
            if bid < 0:
                raise RuntimeError(f"CMJ body not found for MakeHuman visual: {bname}")
            self.cmj_bid[bname] = int(bid)

    def _site(self, cmj_data: Any, name: str) -> np.ndarray:
        return np.array(cmj_data.site_xpos[self.cmj_sid[name]], dtype=np.float64)

    def _body(self, cmj_data: Any, name: str) -> np.ndarray:
        return np.array(cmj_data.xpos[self.cmj_bid[name]], dtype=np.float64)

    @staticmethod
    def _quat_from_to(local_axis: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
        a = np.asarray(local_axis, dtype=np.float64)
        b = np.asarray(target_axis, dtype=np.float64)
        an = np.linalg.norm(a)
        bn = np.linalg.norm(b)
        if an <= 1e-12 or bn <= 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        a = a / an
        b = b / bn
        dot = min(1.0, max(-1.0, float(np.dot(a, b))))
        if dot > 0.999999:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if dot < -0.999999:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(a, ref))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = np.cross(a, ref)
            axis /= max(np.linalg.norm(axis), 1e-12)
            return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)
        axis = np.cross(a, b)
        q = np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=np.float64)
        q /= max(np.linalg.norm(q), 1e-12)
        return q

    @staticmethod
    def _solve_arm_ik(shoulder: np.ndarray, hand: np.ndarray, side_sign: float) -> np.ndarray:
        upper = VISUAL_ARM_UPPER_LEN_M
        fore = VISUAL_ARM_FORE_LEN_M
        span = np.asarray(hand, dtype=np.float64) - np.asarray(shoulder, dtype=np.float64)
        dist = float(np.linalg.norm(span))
        if dist <= 1e-9:
            return np.asarray(shoulder, dtype=np.float64) + np.array([0.0, side_sign * 0.08, -upper])
        unit = span / dist
        dist_c = min(max(dist, abs(upper - fore) + 1e-4), upper + fore - 1e-4)
        a = (upper * upper - fore * fore + dist_c * dist_c) / (2.0 * dist_c)
        h = math.sqrt(max(upper * upper - a * a, 0.0))
        bend_ref = np.array([-0.25, side_sign * 0.16, -1.0], dtype=np.float64)
        bend = bend_ref - unit * float(np.dot(bend_ref, unit))
        bn = np.linalg.norm(bend)
        if bn <= 1e-9:
            bend = np.array([0.0, side_sign, 0.0], dtype=np.float64)
        else:
            bend /= bn
        return np.asarray(shoulder, dtype=np.float64) + unit * a + bend * h

    def _set_segment(self, name: str, pos: np.ndarray, axis_vec: np.ndarray) -> None:
        adr = self.jadr[name]
        q = self._quat_from_to(MAKEHUMAN_SEGMENT_LOCAL_AXES[name], axis_vec)
        visual_state = self.data.qpos
        np.copyto(visual_state[adr:adr + 3], np.asarray(pos, dtype=np.float64))
        np.copyto(visual_state[adr + 3:adr + 7], q)

    def _retarget(self, cmj_data: Any) -> None:
        pelvis = self._site(cmj_data, "pelvis_site")
        sternum = self._site(cmj_data, "sternum_site")
        head = self._site(cmj_data, "head_site")
        left_hip = self._site(cmj_data, "left_hip_site")
        right_hip = self._site(cmj_data, "right_hip_site")
        left_knee = self._site(cmj_data, "left_knee_site")
        right_knee = self._site(cmj_data, "right_knee_site")
        left_ankle = self._site(cmj_data, "left_ankle_site")
        right_ankle = self._site(cmj_data, "right_ankle_site")
        left_heel = self._site(cmj_data, "left_heel_site")
        right_heel = self._site(cmj_data, "right_heel_site")
        left_toe = self._site(cmj_data, "left_toe_site")
        right_toe = self._site(cmj_data, "right_toe_site")
        left_shoulder = self._body(cmj_data, "left_upper_arm")
        right_shoulder = self._body(cmj_data, "right_upper_arm")
        left_hand = self._site(cmj_data, "left_bar_grip_site")
        right_hand = self._site(cmj_data, "right_bar_grip_site")
        left_elbow = self._solve_arm_ik(left_shoulder, left_hand, 1.0)
        right_elbow = self._solve_arm_ik(right_shoulder, right_hand, -1.0)
        self.last_left_hand = left_hand
        self.last_right_hand = right_hand

        trunk_axis = sternum - pelvis
        self._set_segment("mh_pelvis", pelvis + np.array([0.0, 0.0, 0.015]), trunk_axis)
        self._set_segment("mh_torso", 0.5 * (pelvis + sternum), trunk_axis)
        self._set_segment("mh_head", head, head - sternum)
        self._set_segment("mh_left_thigh", 0.5 * (left_hip + left_knee), left_knee - left_hip)
        self._set_segment("mh_right_thigh", 0.5 * (right_hip + right_knee), right_knee - right_hip)
        self._set_segment("mh_left_shank", 0.5 * (left_knee + left_ankle), left_ankle - left_knee)
        self._set_segment("mh_right_shank", 0.5 * (right_knee + right_ankle), right_ankle - right_knee)
        left_foot_center = 0.5 * (left_heel + left_toe)
        right_foot_center = 0.5 * (right_heel + right_toe)
        # The MakeHuman foot mesh is centered about 10 cm above its sole. The
        # plant heel/toe sites are contact landmarks near the deck, so the visual
        # mesh center is raised by the same amount; this keeps the sole on the
        # plate instead of buried in it while leaving contact truth unchanged.
        left_foot_center[2] = max(float(left_heel[2]), float(left_toe[2])) + 0.105
        right_foot_center[2] = max(float(right_heel[2]), float(right_toe[2])) + 0.105
        self._set_segment("mh_left_foot", left_foot_center, left_toe - left_heel)
        self._set_segment("mh_right_foot", right_foot_center, right_toe - right_heel)
        self._set_segment("mh_left_upper_arm", 0.5 * (left_shoulder + left_elbow), left_elbow - left_shoulder)
        self._set_segment("mh_right_upper_arm", 0.5 * (right_shoulder + right_elbow), right_elbow - right_shoulder)
        self._set_segment("mh_left_forearm", 0.5 * (left_elbow + left_hand), left_hand - left_elbow)
        self._set_segment("mh_right_forearm", 0.5 * (right_elbow + right_hand), right_hand - right_elbow)
        mujoco.mj_forward(self.model, self.data)

    def composite(self, base_rgb: np.ndarray, base_depth: np.ndarray, cmj_data: Any) -> tuple[np.ndarray, bool]:
        """Render the retargeted MakeHuman surface and depth-composite it."""
        if not self.available:
            return base_rgb, False
        self._retarget(cmj_data)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        human_rgb = np.ascontiguousarray(self.renderer.render()[:, :, :3])
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        human_depth = np.asarray(self.renderer.render(), dtype=np.float64)
        self.renderer.disable_depth_rendering()
        human = human_rgb.sum(axis=2) > 12
        show = human & (human_depth <= (np.asarray(base_depth, dtype=np.float64) + 0.02))
        out = base_rgb.copy()
        out[show] = human_rgb[show]
        self.frames_rendered += 1
        return out, True

    def metadata(self) -> dict[str, Any]:
        seg_meta = self.segment_manifest.get("segments", {}) if isinstance(self.segment_manifest, dict) else {}
        return {
            "asset_xml": str(self.asset_xml),
            "asset_obj": str(self.asset_obj),
            "asset_dae": str(self.asset_dae),
            "segment_manifest": str(self.manifest_path),
            "available": bool(self.available),
            "load_error": self.load_error,
            "variant": "MakeHuman repository-local OBJ segmented soft-tissue wrapper",
            "render_only": True,
            "stepped": False,
            "posed_from_live_cmj_state": True,
            "retarget_method": "OBJ-derived anatomical segments driven by exact CMJ sites plus visual-only arm IK to bar grip sites",
            "composite_method": "separate MuJoCo render + metric depth compositing over plant frame",
            "obj_used": bool(self.asset_obj.exists()),
            "dae_vendored": bool(self.asset_dae.exists()),
            "segment_count": len(seg_meta),
            "segments": sorted(seg_meta),
            "visual_arm_ik": True,
            "frames_rendered": int(self.frames_rendered),
        }


class _MakeHumanSkinRenderer:
    """PRIMARY render-only human: a true MakeHuman LBS skin.

    Loads the MuJoCo-native ``<skin>`` asset generated offline from the vendored
    MakeHuman DAE. The skin is one continuous soft-tissue surface bound to 163
    render-only mocap "bone" bodies. Every rendered frame those bone bodies are
    posed (mocap_pos/mocap_quat only, then mj_forward for rendering) by a
    piecewise-rigid retarget from the exact live CMJ landmark sites, so MuJoCo's
    native skinning deforms the surface to follow the authoritative plant state.

    This model is physics-inert (no dof, no contacts) and completely isolated
    from the dynamics plant/data generator; it only ever *reads* CMJ
    site_xpos / body xpos.
    """

    # sid -> segment key. bone bodies whose sid is not listed fall back to trunk.
    _Z = np.array([0.0, 0.0, 1.0])
    _Y = np.array([0.0, 1.0, 0.0])

    def __init__(self, task_dir: Path, width: int, height: int, cam_lookat,
                 cam_distance: float, cam_azimuth: float, cam_elevation: float) -> None:
        self.asset_xml = (task_dir / MAKEHUMAN_SKIN_RELATIVE_XML).resolve()
        self.bones_json = (task_dir / MAKEHUMAN_SKIN_BONES_RELATIVE_JSON).resolve()
        self.asset_dae = (task_dir / MAKEHUMAN_DAE_RELATIVE_PATH).resolve()
        self.width = int(width)
        self.height = int(height)
        self.available = False
        self.load_error: str | None = None
        self.frames_rendered = 0
        self.model: Any = None
        self.data: Any = None
        self.renderer: Any = None
        self.camera: Any = None
        self.option: Any = None
        self.cmj_sid: dict[str, int] = {}
        self.cmj_bid: dict[str, int] = {}
        self.n_bones = 0
        self.last_left_hand: np.ndarray | None = None
        self.last_right_hand: np.ndarray | None = None

        missing = [str(p) for p in (self.asset_xml, self.bones_json) if not p.exists()]
        if missing:
            self.load_error = "missing MakeHuman skin asset(s): " + ", ".join(missing)
            return
        try:
            bones = json.loads(self.bones_json.read_text(encoding="utf-8"))["bones"]
            self.sids = [b["sid"] for b in bones]
            self._sid_index = {s: i for i, s in enumerate(self.sids)}
            self.n_bones = len(bones)
            self.bind_pos = np.array([b["bind_pos"] for b in bones], dtype=np.float64)
            self.bind_quat = np.array([b["bind_quat"] for b in bones], dtype=np.float64)
            self._bind_by_sid = {b["sid"]: np.array(b["bind_pos"], dtype=np.float64)
                                 for b in bones}
            self.model = mujoco.MjModel.from_xml_path(str(self.asset_xml))
            self.data = mujoco.MjData(self.model)
            self.model.vis.global_.offwidth = self.width
            self.model.vis.global_.offheight = self.height
            self.model.vis.headlight.ambient[:] = [0.56, 0.56, 0.56]
            self.model.vis.headlight.diffuse[:] = [0.62, 0.62, 0.62]
            self.model.vis.headlight.specular[:] = [0.08, 0.08, 0.08]
            # qpos address of each bone body's free joint (render-only visual model;
            # JSON order matches worldbody order). The driver writes these qpos
            # slices then mj_forward's for rendering, exactly like the segmented
            # visual model -- never the dynamics plant.
            self.qadr = np.empty(self.n_bones, dtype=int)
            for i, b in enumerate(bones):
                jid = mujoco.mj_name2id(self.model, int(mujoco.mjtObj.mjOBJ_JOINT),
                                        f"{b['body']}_free")
                if jid < 0:
                    raise RuntimeError(f"skin bone free joint not found: {b['body']}_free")
                self.qadr[i] = int(self.model.jnt_qposadr[jid])
            self._build_segments()
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = list(cam_lookat)
            cam.distance = float(cam_distance)
            cam.azimuth = float(cam_azimuth)
            cam.elevation = float(cam_elevation)
            self.camera = cam
            opt = mujoco.MjvOption()
            opt.sitegroup[:] = 0
            opt.flags[int(mujoco.mjtVisFlag.mjVIS_TENDON)] = 0
            self.option = opt
            self.available = True
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.available = False

    # -- segment definitions ------------------------------------------------- #
    def _classify(self, sid: str) -> str:
        def side(s):
            return "_L" if s.endswith("_L") else ("_R" if s.endswith("_R") else "")
        sd = side(sid)
        if sid.startswith("upperleg"):
            return "thigh" + sd
        if sid.startswith("lowerleg"):
            return "shank" + sd
        if sid.startswith("foot") or sid.startswith("toe"):
            return "foot" + sd
        if sid.startswith("upperarm"):
            return "uarm" + sd
        if sid.startswith(("lowerarm", "wrist", "finger", "metacarpal")):
            return "farm" + sd
        if sid.startswith(("neck", "head", "jaw", "eye", "oculi", "oris", "levator",
                           "temporalis", "risorius", "orbicularis", "special", "tongue")):
            return "head"
        # root, spine*, pelvis_L/R, breast*, clavicle*, shoulder01* and anything
        # else ride rigidly with the trunk.
        return "trunk"

    def _build_segments(self) -> None:
        # membership: segment key -> list of bone indices
        self.seg_members: dict[str, list[int]] = {}
        for i, sid in enumerate(self.sids):
            self.seg_members.setdefault(self._classify(sid), []).append(i)
        bp = self._bind_by_sid

        def bind_frame(prox_sid, dist_sid, ref):
            p0 = bp[prox_sid]
            p1 = bp[dist_sid]
            return p0, p1, self._frame(p1 - p0, ref)

        Z, Y = self._Z, self._Y
        # (bind proximal joint, bind distal joint, bind anterior/up reference)
        self.seg_bind = {
            "trunk": bind_frame("root", "spine01", Z),
            "head": bind_frame("spine01", "head", Z),
            "uarm_L": bind_frame("upperarm01_L", "lowerarm01_L", Z),
            "uarm_R": bind_frame("upperarm01_R", "lowerarm01_R", Z),
            "farm_L": bind_frame("lowerarm01_L", "wrist_L", Z),
            "farm_R": bind_frame("lowerarm01_R", "wrist_R", Z),
            "thigh_L": bind_frame("upperleg01_L", "lowerleg01_L", Z),
            "thigh_R": bind_frame("upperleg01_R", "lowerleg01_R", Z),
            "shank_L": bind_frame("lowerleg01_L", "foot_L", Z),
            "shank_R": bind_frame("lowerleg01_R", "foot_R", Z),
            "foot_L": bind_frame("foot_L", "toe1-1_L", Y),
            "foot_R": bind_frame("foot_R", "toe1-1_R", Y),
        }

    @staticmethod
    def _frame(primary: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Right-handed 3x3 (columns x,y,z) with z along ``primary`` and x in the
        ``ref`` direction (Gram-Schmidt). Robust to near-parallel primary/ref."""
        z = np.asarray(primary, dtype=np.float64)
        n = np.linalg.norm(z)
        if n <= 1e-12:
            return np.eye(3)
        z = z / n
        r = np.asarray(ref, dtype=np.float64)
        x = r - z * float(np.dot(r, z))
        if np.linalg.norm(x) <= 1e-9:
            alt = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x = alt - z * float(np.dot(alt, z))
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.column_stack([x, y, z])

    def bind_cmj(self, cmj_model: Any) -> None:
        for sname in (
            "pelvis_site", "sternum_site", "head_site",
            "left_hip_site", "right_hip_site", "left_knee_site", "right_knee_site",
            "left_ankle_site", "right_ankle_site", "left_heel_site", "right_heel_site",
            "left_toe_site", "right_toe_site", "left_bar_grip_site", "right_bar_grip_site",
        ):
            sid = mujoco.mj_name2id(cmj_model, int(mujoco.mjtObj.mjOBJ_SITE), sname)
            if sid < 0:
                raise RuntimeError(f"CMJ site not found for MakeHuman skin: {sname}")
            self.cmj_sid[sname] = int(sid)
        for bname in ("left_upper_arm", "right_upper_arm"):
            bid = mujoco.mj_name2id(cmj_model, int(mujoco.mjtObj.mjOBJ_BODY), bname)
            if bid < 0:
                raise RuntimeError(f"CMJ body not found for MakeHuman skin: {bname}")
            self.cmj_bid[bname] = int(bid)

    def _site(self, d: Any, name: str) -> np.ndarray:
        return np.array(d.site_xpos[self.cmj_sid[name]], dtype=np.float64)

    def _body(self, d: Any, name: str) -> np.ndarray:
        return np.array(d.xpos[self.cmj_bid[name]], dtype=np.float64)

    def _pose_segment(self, key: str, live_prox: np.ndarray, live_dist: np.ndarray,
                      live_ref: np.ndarray) -> None:
        """Rigidly place every bone of a segment so its bind proximal joint lands
        on ``live_prox`` and its bind axis maps onto ``live_prox->live_dist``,
        with intra-segment offsets scaled to match the live segment length."""
        p0_bind, p1_bind, R_bind = self.seg_bind[key]
        live_axis = np.asarray(live_dist, dtype=np.float64) - np.asarray(live_prox, dtype=np.float64)
        R_live = self._frame(live_axis, live_ref)
        R = R_live @ R_bind.T                       # bind-frame -> live-frame delta
        bind_len = float(np.linalg.norm(p1_bind - p0_bind))
        live_len = float(np.linalg.norm(live_axis))
        s = (live_len / bind_len) if bind_len > 1e-9 else 1.0
        s = min(2.5, max(0.4, s))               # bound visual length scale
        R_quat = np.zeros(4)
        mujoco.mju_mat2Quat(R_quat, np.ascontiguousarray(R.reshape(9)))
        visual_state = self.data.qpos
        for i in self.seg_members.get(key, []):
            pos = np.asarray(live_prox, dtype=np.float64) + R @ (s * (self.bind_pos[i] - p0_bind))
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, R_quat, np.ascontiguousarray(self.bind_quat[i]))
            adr = int(self.qadr[i])
            np.copyto(visual_state[adr:adr + 3], pos)
            np.copyto(visual_state[adr + 3:adr + 7], q)

    def _retarget(self, d: Any) -> None:
        pelvis = self._site(d, "pelvis_site")
        sternum = self._site(d, "sternum_site")
        head = self._site(d, "head_site")
        lhip = self._site(d, "left_hip_site"); rhip = self._site(d, "right_hip_site")
        lknee = self._site(d, "left_knee_site"); rknee = self._site(d, "right_knee_site")
        lank = self._site(d, "left_ankle_site"); rank = self._site(d, "right_ankle_site")
        lheel = self._site(d, "left_heel_site"); rheel = self._site(d, "right_heel_site")
        ltoe = self._site(d, "left_toe_site"); rtoe = self._site(d, "right_toe_site")
        lhand = self._site(d, "left_bar_grip_site"); rhand = self._site(d, "right_bar_grip_site")
        self.last_left_hand = lhand
        self.last_right_hand = rhand

        # World anterior (facing) direction from the feet; up is +Z.
        ant = 0.5 * (ltoe + rtoe) - 0.5 * (lheel + rheel)
        ant[2] = 0.0
        if np.linalg.norm(ant) < 1e-6:
            ant = np.array([1.0, 0.0, 0.0])
        ant /= np.linalg.norm(ant)
        up = self._Z

        self._pose_segment("trunk", pelvis, sternum, ant)
        self._pose_segment("head", sternum, head, ant)
        self._pose_segment("thigh_L", lhip, lknee, ant)
        self._pose_segment("thigh_R", rhip, rknee, ant)
        self._pose_segment("shank_L", lknee, lank, ant)
        self._pose_segment("shank_R", rknee, rank, ant)
        # Feet: primary along ankle->toe, roll reference is world up so soles stay
        # down. The visual foot is compacted toward the ankle (see FOOT_VISUAL_*)
        # so the toe mesh reads as a human foot rather than a stretched flipper;
        # contact truth is untouched (heel/toe sites and their markers do not move).
        foot_drop = np.array([0.0, 0.0, FOOT_VISUAL_SOLE_DROP_M])
        self._pose_segment("foot_L", lank - foot_drop,
                           lank + FOOT_VISUAL_LEN_SCALE * (ltoe - lank) - foot_drop, up)
        self._pose_segment("foot_R", rank - foot_drop,
                           rank + FOOT_VISUAL_LEN_SCALE * (rtoe - rank) - foot_drop, up)
        # Arms + gripping hands: anatomical shoulder->elbow->wrist IK to the exact
        # bar grip sites, with the real MakeHuman finger bones curled around the
        # shaft. Overwrites only the arm/hand bones (trunk/legs/feet untouched).
        self._pose_arms_and_hands(d, ant)
        mujoco.mj_forward(self.model, self.data)

    # -- arms + gripping hands ----------------------------------------------- #
    @staticmethod
    def _rotmat(axis: np.ndarray, ang: float) -> np.ndarray:
        """3x3 right-handed rotation of ``ang`` rad about ``axis`` (Rodrigues)."""
        axis = np.asarray(axis, dtype=np.float64)
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            return np.eye(3)
        x, y, z = axis / n
        c = math.cos(ang); s = math.sin(ang); C = 1.0 - c
        return np.array([
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ], dtype=np.float64)

    @staticmethod
    def _solve_elbow(shoulder: np.ndarray, wrist: np.ndarray, upper: float,
                     fore: float, bend_ref: np.ndarray) -> np.ndarray:
        """Planar 2-link elbow between ``shoulder`` and ``wrist`` (upper/fore link
        lengths), placed in the half-plane picked by ``bend_ref``."""
        span = np.asarray(wrist, dtype=np.float64) - np.asarray(shoulder, dtype=np.float64)
        dist = float(np.linalg.norm(span))
        if dist <= 1e-9:
            return np.asarray(shoulder, dtype=np.float64) + np.array([0.0, 0.0, -upper])
        unit = span / dist
        dist_c = min(max(dist, abs(upper - fore) + 1e-4), upper + fore - 1e-4)
        a = (upper * upper - fore * fore + dist_c * dist_c) / (2.0 * dist_c)
        h = math.sqrt(max(upper * upper - a * a, 0.0))
        bend = np.asarray(bend_ref, dtype=np.float64)
        bend = bend - unit * float(np.dot(bend, unit))
        nb = float(np.linalg.norm(bend))
        bend = bend / nb if nb > 1e-9 else np.array([0.0, 0.0, -1.0])
        return np.asarray(shoulder, dtype=np.float64) + unit * a + bend * h

    def _set_bone(self, idx: int, pos: np.ndarray, R: np.ndarray) -> None:
        """Write one render-only skin bone's free-joint qpos from a world pose."""
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.ascontiguousarray(R.reshape(9)))
        qq = np.zeros(4)
        mujoco.mju_mulQuat(qq, q, np.ascontiguousarray(self.bind_quat[idx]))
        adr = int(self.qadr[idx])
        np.copyto(self.data.qpos[adr:adr + 3], np.asarray(pos, dtype=np.float64))
        np.copyto(self.data.qpos[adr + 3:adr + 7], qq)

    def _place_bones(self, idxs, bp0, bp1, bref, lp0, lp1, lref,
                     srange: tuple[float, float] = (0.4, 2.5)) -> np.ndarray:
        """Rigidly map bind segment ``bp0->bp1`` onto live ``lp0->lp1`` for every
        bone in ``idxs`` (same construction as ``_pose_segment`` but with explicit
        bind endpoints, so sub-groups of a limb can be posed independently)."""
        Rb = self._frame(bp1 - bp0, bref)
        Rl = self._frame(lp1 - lp0, lref)
        R = Rl @ Rb.T
        bl = float(np.linalg.norm(bp1 - bp0))
        ll = float(np.linalg.norm(lp1 - lp0))
        s = min(srange[1], max(srange[0], ll / bl if bl > 1e-9 else 1.0))
        for i in idxs:
            self._set_bone(i, np.asarray(lp0, dtype=np.float64) + R @ (s * (self.bind_pos[i] - bp0)), R)
        return R

    # per-finger MCP/PIP/DIP flexion (rad); tuned to wrap a ~7 cm shaft
    _FINGER_CURL = {2: (0.62, 0.85, 0.60), 3: (0.66, 0.90, 0.62),
                    4: (0.66, 0.90, 0.62), 5: (0.62, 0.85, 0.58)}
    _THUMB_CURL = (0.50, 0.55, 0.50)

    def _pose_arms_and_hands(self, d: Any, ant: np.ndarray) -> None:
        """Pose both arms and gripping hands from the exact plant shoulder bodies
        and bar grip sites. Anatomical shoulder->elbow->wrist IK plus a cylinder
        grip that curls the real MakeHuman finger bones around the bar shaft.

        Render-only: writes only render-model qpos for the arm/hand skin bones.
        """
        bind = self._bind_by_sid

        def group(prefixes, side):
            return [i for i, s in enumerate(self.sids)
                    if s.endswith(side) and s.startswith(prefixes)]

        for side, sgn, arm_body, grip in (
            ("_L", 1.0, "left_upper_arm", "left_bar_grip_site"),
            ("_R", -1.0, "right_upper_arm", "right_bar_grip_site"),
        ):
            uarm = group(("upperarm",), side)
            farm = group(("lowerarm",), side)
            wristmeta = group(("wrist", "metacarpal"), side)
            shoulder = self._body(d, arm_body)
            G = self._site(d, grip)
            u = np.array([0.0, 1.0, 0.0]) * sgn          # shaft axis, outboard
            # First pass: rough elbow -> forearm approach direction to the shaft.
            e0 = self._solve_elbow(shoulder, G, 0.237, 0.239,
                                   np.array([-0.25, sgn * 0.16, -1.0]))
            appr = G - e0
            na = float(np.linalg.norm(appr))
            appr = appr / na if na > 1e-9 else np.array([0.0, 0.0, 1.0])
            wrist = G - appr * 0.085                      # wrist sits a hand back from shaft
            elbow = self._solve_elbow(shoulder, wrist, 0.237, 0.150,
                                      np.array([-0.2, sgn * 0.12, -1.0]))
            # Upper arm (shoulder->elbow) and forearm (elbow->wrist).
            self._place_bones(uarm, bind["upperarm01" + side], bind["lowerarm01" + side],
                              self._Z, shoulder, elbow, ant)
            self._place_bones(farm, bind["lowerarm01" + side], bind["wrist" + side],
                              self._Z, elbow, wrist, ant)
            # Rigid wrist + metacarpals (palm/back of hand): bind wrist->mid-knuckle
            # onto live wrist->shaft, so the palm seats against the bar.
            Rh = self._place_bones(wristmeta, bind["wrist" + side], bind["finger3-1" + side],
                                   self._Z, wrist, G, ant, srange=(0.5, 1.8))
            sh = min(1.8, max(0.5, float(np.linalg.norm(G - wrist))
                              / max(1e-9, float(np.linalg.norm(bind["finger3-1" + side] - bind["wrist" + side])))))

            def hand_world(bp):
                return wrist + Rh @ (sh * (bp - bind["wrist" + side]))

            e1 = ant - u * float(np.dot(ant, u))          # anterior, perp to shaft
            e1 = e1 / max(1e-9, float(np.linalg.norm(e1)))
            # Long fingers: articulated per-phalange curl about the shaft axis from each knuckle.
            for j, angs in self._FINGER_CURL.items():
                fb = ["finger%d-1%s" % (j, side), "finger%d-2%s" % (j, side),
                      "finger%d-3%s" % (j, side)]
                idxs = [self._sid_index[s] for s in fb]
                K = hand_world(bind[fb[0]])
                tip_b = bind[fb[2]] + (bind[fb[2]] - bind[fb[1]])
                dir0 = hand_world(tip_b) - K
                dir0 = dir0 / max(1e-9, float(np.linalg.norm(dir0)))
                seg_len = [
                    float(np.linalg.norm(hand_world(bind[fb[1]]) - K)),
                    float(np.linalg.norm(hand_world(bind[fb[2]]) - hand_world(bind[fb[1]]))),
                    float(np.linalg.norm(hand_world(tip_b) - hand_world(bind[fb[2]]))),
                ]

                def tip_radius(sign):
                    p = K.copy(); cum = 0.0
                    for k in range(3):
                        cum += angs[k]
                        p = p + seg_len[k] * (self._rotmat(u, sign * cum) @ dir0)
                    rad = (p - G) - u * float(np.dot(p - G, u))
                    return float(np.linalg.norm(rad))

                sign = 1.0 if tip_radius(1.0) < tip_radius(-1.0) else -1.0
                p0 = K.copy(); cum = 0.0
                for k in range(3):
                    cum += angs[k]
                    p1 = p0 + seg_len[k] * (self._rotmat(u, sign * cum) @ dir0)
                    bp1 = bind[fb[k + 1]] if k < 2 else tip_b
                    self._place_bones([idxs[k]], bind[fb[k]], bp1, u, p0, p1, u,
                                      srange=(0.4, 2.2))
                    p0 = p1
            # Thumb: opposes the fingers, wrapping across the shaft.
            tb = ["finger1-1" + side, "finger1-2" + side, "finger1-3" + side]
            tidx = [self._sid_index[s] for s in tb]
            K = hand_world(bind[tb[0]])
            tip_b = bind[tb[2]] + (bind[tb[2]] - bind[tb[1]])
            dir0 = hand_world(tip_b) - K
            dir0 = dir0 / max(1e-9, float(np.linalg.norm(dir0)))
            seg_len = [
                float(np.linalg.norm(hand_world(bind[tb[1]]) - K)),
                float(np.linalg.norm(hand_world(bind[tb[2]]) - hand_world(bind[tb[1]]))),
                float(np.linalg.norm(hand_world(tip_b) - hand_world(bind[tb[2]]))),
            ]
            taxis = np.cross(u, e1)                        # ~ up, thumb crosses the shaft

            def thumb_radius(sign):
                p = K.copy(); cum = 0.0
                for k in range(3):
                    cum += self._THUMB_CURL[k]
                    p = p + seg_len[k] * (self._rotmat(taxis, sign * cum) @ dir0)
                rad = (p - G) - u * float(np.dot(p - G, u))
                return float(np.linalg.norm(rad))

            ts = 1.0 if thumb_radius(1.0) < thumb_radius(-1.0) else -1.0
            p0 = K.copy(); cum = 0.0
            for k in range(3):
                cum += self._THUMB_CURL[k]
                p1 = p0 + seg_len[k] * (self._rotmat(taxis, ts * cum) @ dir0)
                bp1 = bind[tb[k + 1]] if k < 2 else tip_b
                self._place_bones([tidx[k]], bind[tb[k]], bp1, taxis, p0, p1, taxis,
                                  srange=(0.4, 2.2))
                p0 = p1

    def composite(self, base_rgb: np.ndarray, base_depth: np.ndarray, cmj_data: Any) -> tuple[np.ndarray, bool]:
        """Render the retargeted skin and depth-composite it over the plant frame."""
        if not self.available:
            return base_rgb, False
        self._retarget(cmj_data)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        human_rgb = np.ascontiguousarray(self.renderer.render()[:, :, :3])
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        human_depth = np.asarray(self.renderer.render(), dtype=np.float64)
        self.renderer.disable_depth_rendering()
        human = human_rgb.sum(axis=2) > 12
        show = human & (human_depth <= (np.asarray(base_depth, dtype=np.float64) + 0.02))
        out = base_rgb.copy()
        out[show] = human_rgb[show]
        self.frames_rendered += 1
        return out, True

    def metadata(self) -> dict[str, Any]:
        return {
            "asset_xml": str(self.asset_xml),
            "asset_dae": str(self.asset_dae),
            "bones_json": str(self.bones_json),
            "available": bool(self.available),
            "load_error": self.load_error,
            "variant": "MakeHuman MuJoCo-native LBS skin (true continuous soft-tissue surface)",
            "render_only": True,
            "stepped": False,
            "posed_from_live_cmj_state": True,
            "primary_human_render": "mujoco_native_skin",
            "mujoco_native_skin_used": True,
            "rigged_makehuman_scene_skin_used": True,
            "segmented_obj_primary_human": False,
            "procedural_capsule_human": False,
            "makehuman_obj_used": False,
            "n_skin_bones": int(self.n_bones),
            "retarget_method": (
                "piecewise-rigid per-segment LBS bone retarget from exact CMJ "
                "landmark sites (+ visual-only arm IK to bar grip sites); free-joint "
                "qpos pose writes on render-only skin model, mj_forward for render only"
            ),
            "arm_bar_grip_retargeted": True,
            "finger_grip_mode": "makehuman_bones",
            "hand_finger_bones_curled": True,
            "n_hand_finger_bones_per_side": 20,
            "grip_method": (
                "anatomical shoulder->elbow->wrist 2-link IK to exact bar grip sites; "
                "real MakeHuman wrist/metacarpal/finger bones curled around the bar "
                "shaft via a per-phalange articulated cylinder-wrap grip (render-only)"
            ),
            "composite_method": "separate MuJoCo skin render + metric depth compositing over plant frame",
            "dae_vendored": bool(self.asset_dae.exists()),
            "frames_rendered": int(self.frames_rendered),
        }


def _add_barbell(scene: Any, data: Any, bar_body_id: int) -> None:
    """Draw a 20 kg Olympic bar anchored to the exact live bar body.

    The physical task load is a 20 kg bar. The visual therefore emphasizes the
    full-length steel shaft, sleeves, collars, and end caps rather than adding
    large bumper plates that would imply a different load and obscure the torso.
    """
    if bar_body_id < 0:
        return
    center = np.array(data.xpos[bar_body_id], dtype=np.float64)
    axis = np.array(data.xmat[bar_body_id], dtype=np.float64).reshape(3, 3)[:, 1]
    n = np.linalg.norm(axis)
    if n <= 1e-9:
        return
    axis = axis / n
    # Full-length bar and sleeves. The XML bar body supplies the authoritative
    # pose; these visual dimensions prevent the sagittal projection from reading
    # as a small toy puck.
    _add_segment(scene, center - 1.10 * axis, center + 1.10 * axis, 0.012,
                 BAR_STEEL_RGBA, kind=int(mujoco.mjtGeom.mjGEOM_CYLINDER))
    for sgn in (1.0, -1.0):
        sleeve_in = center + sgn * 0.66 * axis
        sleeve_out = center + sgn * 1.10 * axis
        _add_segment(scene, sleeve_in, sleeve_out, 0.025, BAR_STEEL_RGBA,
                     kind=int(mujoco.mjtGeom.mjGEOM_CYLINDER))
        collar_in = center + sgn * 0.59 * axis
        collar_out = center + sgn * 0.66 * axis
        _add_segment(scene, collar_in, collar_out, 0.035, BAR_COLLAR_RGBA,
                     kind=int(mujoco.mjtGeom.mjGEOM_CYLINDER))
        _add_sphere(scene, center + sgn * 1.105 * axis, 0.027, BAR_STEEL_RGBA, emission=0.08)


def _site_pos(data: Any, site_ids: dict[str, int], name: str) -> np.ndarray | None:
    sid = site_ids.get(name)
    if sid is None:
        return None
    return np.array(data.site_xpos[sid], dtype=np.float64)


def _body_pos(data: Any, body_ids: dict[str, int], name: str) -> np.ndarray | None:
    bid = body_ids.get(name)
    if bid is None:
        return None
    return np.array(data.xpos[bid], dtype=np.float64)


def _add_rigged_makehuman_skin(
    scene: Any,
    data: Any,
    site_ids: dict[str, int],
    body_ids: dict[str, int],
) -> None:
    """Single render-only athlete skin driven by exact plant skeleton sites.

    This replaces the previous separate segmented-MakeHuman compositing path.
    The visual is one coherent rig in the same MuJoCo scene, so no second body can
    depth-composite over the true plant state. It remains render-only: no qpos,
    qvel, controls, contacts, or forces are written.
    """
    skin = MAKEHUMAN_SKIN_RGBA
    skin_dark = (0.63, 0.43, 0.32, 1.0)
    shoe = (0.035, 0.037, 0.042, 1.0)

    pelvis = _site_pos(data, site_ids, "pelvis_site")
    sternum = _site_pos(data, site_ids, "sternum_site")
    head = _site_pos(data, site_ids, "head_site")
    if pelvis is not None and sternum is not None:
        _add_segment(scene, pelvis + np.array([0.0, 0.0, -0.035]),
                     sternum + np.array([-0.015, 0.0, 0.020]), 0.118, skin)
        _add_segment(scene, pelvis + np.array([0.0, 0.0, -0.060]),
                     pelvis + np.array([0.0, 0.0, 0.055]), 0.145, skin_dark)
    if sternum is not None and head is not None:
        _add_segment(scene, sternum + np.array([-0.010, 0.0, 0.055]),
                     head + np.array([-0.010, 0.0, -0.105]), 0.040, skin)
        _add_sphere(scene, head + np.array([0.012, 0.0, 0.0]), 0.092, skin, emission=0.10)

    for side, sign in (("left", 1.0), ("right", -1.0)):
        hip = _site_pos(data, site_ids, f"{side}_hip_site")
        knee = _site_pos(data, site_ids, f"{side}_knee_site")
        ankle = _site_pos(data, site_ids, f"{side}_ankle_site")
        heel = _site_pos(data, site_ids, f"{side}_heel_site")
        toe = _site_pos(data, site_ids, f"{side}_toe_site")
        shoulder = _body_pos(data, body_ids, f"{side}_upper_arm")
        hand = _site_pos(data, site_ids, f"{side}_bar_grip_site")
        if hip is not None and knee is not None:
            _add_segment(scene, hip, knee, 0.058, skin)
        if knee is not None and ankle is not None:
            _add_segment(scene, knee, ankle, 0.047, skin)
        if heel is not None and toe is not None:
            foot_axis = toe - heel
            mid = 0.5 * (heel + toe)
            _add_segment(scene, mid - 0.40 * foot_axis, mid + 0.58 * foot_axis, 0.034, shoe)
        if shoulder is not None and hand is not None:
            elbow = _MakeHumanSegmentRenderer._solve_arm_ik(shoulder, hand, sign)
            _add_segment(scene, shoulder, elbow, 0.032, skin)
            _add_segment(scene, elbow, hand, 0.026, skin)
            _add_sphere(scene, hand, 0.018, skin, emission=0.08)


def _add_makehuman_mesh_seam_caps(
    scene: Any,
    data: Any,
    site_ids: dict[str, int],
    body_ids: dict[str, int],
) -> None:
    """Small render-only caps that close segmented MakeHuman joint seams.

    These are not structural plant limbs and do not duplicate the body. They only
    cover small discontinuities introduced by retargeting separate MakeHuman OBJ
    segments to a different MuJoCo skeleton.
    """
    skin = MAKEHUMAN_SKIN_RGBA
    for name, radius in (
        ("pelvis_site", 0.040),
        ("sternum_site", 0.030),
        ("left_hip_site", 0.040), ("right_hip_site", 0.040),
        ("left_knee_site", 0.032), ("right_knee_site", 0.032),
        ("left_ankle_site", 0.026), ("right_ankle_site", 0.026),
    ):
        p = _site_pos(data, site_ids, name)
        if p is not None:
            _add_sphere(scene, p, radius, skin, emission=0.05)

    sternum = _site_pos(data, site_ids, "sternum_site")
    head = _site_pos(data, site_ids, "head_site")
    if sternum is not None and head is not None:
        neck0 = sternum + 0.30 * (head - sternum)
        neck1 = sternum + 0.58 * (head - sternum)
        _add_segment(scene, neck0, neck1, 0.026, skin)

def _add_clean_bar_limb_overlays(
    scene: Any,
    data: Any,
    site_ids: dict[str, int],
) -> None:
    """Clean render-only arms and shoes tied to exact bar/foot landmarks."""
    skin = MAKEHUMAN_SKIN_RGBA
    shoe = (0.035, 0.037, 0.042, 1.0)
    sternum = _site_pos(data, site_ids, "sternum_site")
    pelvis = _site_pos(data, site_ids, "pelvis_site")
    if sternum is not None:
        for side, sign in (("left", 1.0), ("right", -1.0)):
            hand = _site_pos(data, site_ids, f"{side}_bar_grip_site")
            if hand is not None:
                shoulder = sternum + np.array([-0.035, sign * 0.175, 0.010], dtype=np.float64)
                elbow = _MakeHumanSegmentRenderer._solve_arm_ik(shoulder, hand, sign)
                _add_sphere(scene, shoulder, 0.034, skin, emission=0.05)
                _add_segment(scene, shoulder, elbow, 0.027, skin)
                _add_segment(scene, elbow, hand, 0.023, skin)
                _add_sphere(scene, hand, 0.018, skin, emission=0.04)
    for side in ("left", "right"):
        heel = _site_pos(data, site_ids, f"{side}_heel_site")
        toe = _site_pos(data, site_ids, f"{side}_toe_site")
        if heel is None or toe is None:
            continue
        foot_axis = toe - heel
        mid = 0.5 * (heel + toe)
        mid[2] = max(float(heel[2]), float(toe[2])) + 0.038
        p0 = mid - 0.42 * foot_axis
        p1 = mid + 0.58 * foot_axis
        p0[2] = mid[2]
        p1[2] = mid[2]
        _add_segment(scene, p0, p1, 0.034, shoe)


def _add_segment(scene: Any, p0, p1, radius: float, rgba, kind: int | None = None) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    geom_kind = int(mujoco.mjtGeom.mjGEOM_CAPSULE if kind is None else kind)
    mujoco.mjv_initGeom(
        g, geom_kind, np.zeros(3), np.zeros(3), np.eye(3).reshape(9),
        np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(g, geom_kind, radius,
                         np.array(p0, dtype=np.float64), np.array(p1, dtype=np.float64))
    scene.ngeom += 1


def _add_com_path(scene: Any, points: list, rgba) -> None:
    if len(points) < 2:
        return
    step = max(1, len(points) // 100)
    thinned = points[::step]
    if thinned and (thinned[-1] is not points[-1]):
        thinned.append(points[-1])
    for a, b in zip(thinned[:-1], thinned[1:]):
        _add_segment(scene, a, b, 0.010, rgba)
    ghost = (rgba[0], rgba[1], rgba[2], 0.5)
    for p in thinned[::4]:
        _add_sphere(scene, p, 0.013, ghost)


def _add_trail(scene: Any, points: list, rgba, radius: float = 0.006, max_points: int = 0,
               target_segments: int = 44) -> None:
    """Draw a fading polyline through recent exact-state points (newest = brightest).

    Subsampled to <= target_segments so long histories stay within the scene geom
    budget and read cleanly. max_points limits to a recent window (0 = full path).
    """
    pts = points[-max_points:] if max_points and len(points) > max_points else list(points)
    if len(pts) < 2:
        return
    step = max(1, len(pts) // target_segments)
    pts = pts[::step]
    n = len(pts)
    for i in range(1, n):
        frac = i / n
        col = (rgba[0], rgba[1], rgba[2], max(0.08, rgba[3] * frac))
        _add_segment(scene, pts[i - 1], pts[i], radius * (0.4 + 0.6 * frac), col)


def _add_plate_tile(scene: Any, x: float, y: float, top_z: float, rgba) -> None:
    """A faint flat sensor-zone tile on the force-plate top under a foot landmark."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, int(mujoco.mjtGeom.mjGEOM_BOX),
        np.array([0.075, 0.055, 0.003], dtype=np.float64),
        np.array([x, y, top_z + 0.004], dtype=np.float64),
        np.eye(3).reshape(9), np.array(rgba, dtype=np.float32))
    scene.ngeom += 1


def _add_lpt_device(scene: Any, device_pos, bar_lpt_pos, front_sign: float) -> np.ndarray:
    """The single ground LPT measurement device: one compact black housing sat on
    the floor IN FRONT of the force plate, with subtle teal/cyan accent lines and a
    thin near-vertical measurement string that runs UP from the device spool to the
    exact bar attachment site. A string cannot push, so this can never read as an
    overhead support cable; it is measurement-only (0 N). Foot-sized hardware.
    Anchored to a fixed floor spot under the bar; render-only, no physics."""
    base = np.array(device_pos, dtype=np.float64)
    hx, hy, hz = 0.060, 0.040, 0.030  # low-profile foot-scale half-extents
    body_c = base + np.array([0.0, 0.0, hz])
    # Black housing.
    if scene.ngeom < scene.maxgeom:
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g, int(mujoco.mjtGeom.mjGEOM_BOX),
            np.array([hx, hy, hz], dtype=np.float64),
            body_c, np.eye(3).reshape(9), np.array(LPT_BODY_RGBA, dtype=np.float32))
        g.emission = 0.06
        scene.ngeom += 1
    # Subtle cyan accent strip along the top and the camera-facing bezel.
    fy = front_sign * hy
    _add_segment(scene, base + np.array([-hx, fy, 2 * hz]), base + np.array([hx, fy, 2 * hz]),
                 0.006, LPT_ACCENT_RGBA)
    _add_segment(scene, base + np.array([-hx, fy, 0.026]), base + np.array([-hx, fy, 2 * hz - 0.010]),
                 0.005, LPT_ACCENT_RGBA)
    _add_segment(scene, base + np.array([hx, fy, 0.026]), base + np.array([hx, fy, 2 * hz - 0.010]),
                 0.005, LPT_ACCENT_RGBA)
    # Spool/retractor where the string exits, with a cyan indicator ring.
    spool = base + np.array([0.0, front_sign * (hy + 0.010), 2 * hz - 0.010])
    _add_sphere(scene, spool, 0.014, (0.10, 0.11, 0.13, 1.0), emission=0.08)
    _add_sphere(scene, spool + np.array([0.0, front_sign * 0.004, 0.0]), 0.009, LPT_ACCENT_RGBA, emission=0.55)
    # Thin near-vertical measurement string, spool -> exact bar/LPT attach site.
    _add_segment(scene, spool, np.array(bar_lpt_pos, dtype=np.float64), 0.0035,
                 (0.45, 0.86, 0.94, 0.9))
    return spool


def _add_vertical_ruler(scene: Any, x: float, y: float, z0: float, z1: float) -> None:
    rgba_line = (0.55, 0.60, 0.68, 0.9)
    _add_segment(scene, (x, y, z0), (x, y, z1), 0.004, rgba_line)
    n = int(round((z1 - z0) / 0.10))
    for k in range(n + 1):
        z = z0 + k * 0.10
        tick = 0.06 if (k % 5 == 0) else 0.03
        _add_segment(scene, (x, y, z), (x - tick, y, z), 0.004, rgba_line)


def _add_forceplate_zone(scene: Any, model: Any, data: Any, plate_geom: int) -> None:
    """Clean lab force-plate deck at HALF the previous rendered footprint.

    The physical plate geom (half-extents 1.25 x 0.50) is unchanged and omitted for
    rendering; this render-only deck is centred on the true plate top and still sits
    under both feet. The near/far surface zones are truthful visual regions of the
    single aggregate plate; the left/right Fz split comes from real MuJoCo contacts
    with the named left/right foot geoms (not from separate physical sensors).
    """
    pos = np.array(data.geom_xpos[plate_geom], dtype=np.float64)
    plate_size = np.array(model.geom_size[plate_geom], dtype=np.float64)
    top_z = float(pos[2] + max(float(plate_size[2]), 0.01) + 0.002)
    sx, sy = FORCE_PLATE_VISUAL_HALF
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, int(mujoco.mjtGeom.mjGEOM_BOX),
        np.array([sx, sy, 0.018], dtype=np.float64),
        np.array([pos[0], pos[1], top_z - 0.018], dtype=np.float64),
        np.eye(3).reshape(9), np.array((0.18, 0.22, 0.26, 0.98), dtype=np.float32),
    )
    scene.ngeom += 1
    z = float(top_z + 0.004)
    # Two truthful surface zones (camera-near vs far half of the single plate).
    for center_y, rgba in (
        (pos[1] + 0.50 * sy, (0.12, 0.33, 0.42, 0.55)),
        (pos[1] - 0.50 * sy, (0.13, 0.27, 0.38, 0.55)),
    ):
        if scene.ngeom >= scene.maxgeom:
            break
        zg = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            zg, int(mujoco.mjtGeom.mjGEOM_BOX),
            np.array([0.95 * sx, 0.47 * sy, 0.004], dtype=np.float64),
            np.array([pos[0], center_y, z + 0.002], dtype=np.float64),
            np.eye(3).reshape(9), np.array(rgba, dtype=np.float32),
        )
        scene.ngeom += 1
    _add_segment(scene, (pos[0] - sx, pos[1], z + 0.008), (pos[0] + sx, pos[1], z + 0.008),
                 0.004, (0.42, 0.78, 0.84, 0.80))
    edge_rgba = (0.32, 0.70, 0.76, 0.85)
    corners = [
        np.array([pos[0] - sx, pos[1] - sy, z]),
        np.array([pos[0] + sx, pos[1] - sy, z]),
        np.array([pos[0] + sx, pos[1] + sy, z]),
        np.array([pos[0] - sx, pos[1] + sy, z]),
    ]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        _add_segment(scene, a, b, 0.005, edge_rgba)


def _add_soft_tissue_gap_fillers(scene: Any, data: Any, site_ids: dict[str, int],
                                 body_ids: dict[str, int]) -> None:
    """Small skin-toned arm fillers where the segmented visual mesh omits arms."""
    if "sternum_site" not in site_ids:
        return
    skin = (MAKEHUMAN_SKIN_RGBA[0], MAKEHUMAN_SKIN_RGBA[1], MAKEHUMAN_SKIN_RGBA[2], 0.96)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        arm_body = f"{side}_upper_arm"
        grip_site = f"{side}_bar_grip_site"
        if arm_body in body_ids and grip_site in site_ids:
            shoulder = np.array(data.xpos[body_ids[arm_body]], dtype=np.float64)
            hand = np.array(data.site_xpos[site_ids[grip_site]], dtype=np.float64)
            elbow = _MakeHumanSegmentRenderer._solve_arm_ik(shoulder, hand, sign)
            _add_sphere(scene, shoulder, 0.030, skin, emission=0.08)
            _add_segment(scene, shoulder, elbow, 0.027, skin)
            _add_segment(scene, elbow, hand, 0.023, skin)
            _add_sphere(scene, hand, 0.019, skin, emission=0.10)


# --------------------------------------------------------------------------- #
# Rollout recorder: read-only hook around mujoco.mj_step                      #
# --------------------------------------------------------------------------- #
class _RolloutRecorder:
    """Captures exact per-step states and renders live frames of the real rollout.

    The hook delegates to the genuine ``mujoco.mj_step`` first, then only READS
    the resulting state. It writes no qpos/qvel and applies no forces.
    """

    def __init__(self, plant: Any, task_dir: Path, width: int, height: int, settle_steps: int,
                 sample_step_indices: set[int]) -> None:
        self.plant = plant
        self.task_dir = task_dir
        self.width = width
        self.height = height
        self.settle_steps = settle_steps
        self.sample_step_indices = sample_step_indices
        self.step_count = 0
        self.renderer: Any = None
        self.camera: Any = None
        self.scene_option: Any = None
        self.ids: dict[str, int] | None = None
        self.pelvis_body = -1
        self.plate_geom = -1
        self.ndof = 0
        self.pos_samples: list[np.ndarray] = []
        self.vel_samples: list[np.ndarray] = []
        self.root_z_world: list[float] = []
        self.com_points: list[np.ndarray] = []
        self.frames: list[dict[str, Any]] = []
        # Exact-state landmark histories (Slice B trace layer). Filled every
        # post-settle step by reading data.site_xpos; never authored.
        self.joint_site_ids: dict[str, int] = {}
        self.foot_site_ids: dict[str, int] = {}
        self.bar_lpt_site_id = -1
        self.joint_hist: dict[str, list[np.ndarray]] = {}
        self.bar_lpt_points: list[np.ndarray] = []
        self.lpt_device_pos: np.ndarray | None = None
        self.lpt_device_initialized = False
        self.soft_tissue_site_ids: dict[str, int] = {}
        self.soft_tissue_body_ids: dict[str, int] = {}
        # Anatomical landmark layer (exact CMJ sites -> foreground markers/traces).
        self.landmark_site_ids: dict[str, int] = {}
        self.landmark_hist: dict[str, list[np.ndarray]] = {}
        self.bar_body_id = -1
        self.plate_half_y = 0.5
        self.plate_top_z = 0.0
        self.system_mass_kg = 0.0
        self.joint_index_map: dict[str, dict[str, Any]] = {}
        self.makehuman: _MakeHumanSkinRenderer | _MakeHumanSegmentRenderer | None = None
        self.human_is_skin = False
        self.skin_fallback_reason: str | None = None
        self.human_visual_used = False
        self.lpt_quiet_com_x: float | None = None

    def _ensure_renderer(self, model: Any) -> None:
        if self.renderer is not None:
            return
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
        model.vis.headlight.diffuse[:] = [0.55, 0.55, 0.55]
        model.vis.headlight.specular[:] = [0.10, 0.10, 0.10]
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width, max_geom=20000)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = list(CAM_LOOKAT)
        cam.distance = CAM_DISTANCE
        cam.azimuth = CAM_AZIMUTH
        cam.elevation = CAM_ELEVATION
        self.camera = cam
        # Primary human: true MakeHuman MuJoCo-native LBS skin. The older segmented
        # OBJ wrapper is retained only as a disabled fallback if the skin asset is
        # unavailable or fails to bind.
        self.human_is_skin = False
        skin = _MakeHumanSkinRenderer(
            self.task_dir, self.width, self.height,
            CAM_LOOKAT, CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION,
        )
        if skin.available:
            try:
                skin.bind_cmj(model)
                self.makehuman = skin
                self.human_is_skin = True
            except Exception as exc:
                skin.load_error = f"bind failed: {type(exc).__name__}: {exc}"
                skin.available = False
        if not self.human_is_skin:
            self.makehuman = _MakeHumanSegmentRenderer(
                self.task_dir, self.width, self.height,
                CAM_LOOKAT, CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION,
            )
            self.skin_fallback_reason = skin.load_error
            if self.makehuman.available:
                try:
                    self.makehuman.bind_cmj(model)
                except Exception as exc:
                    self.makehuman.load_error = f"bind failed: {type(exc).__name__}: {exc}"
                    self.makehuman.available = False
        opt = mujoco.MjvOption()
        opt.sitegroup[:] = 0  # hide default XML site markers; overlays carry the cues
        # Disable MuJoCo's default spatial-tendon rendering: the lpt_tether tendon
        # (rear anchor at x=-0.80, z=2.10, 0 N) otherwise draws as an overhead
        # support cable. Slice B replaces it with an explicit front measurement
        # device + string. The hand-grip tendons are also omitted (the real arms now
        # visibly hold the bar). Physics is untouched; this is a render flag only.
        opt.flags[int(mujoco.mjtVisFlag.mjVIS_TENDON)] = 0
        self.scene_option = opt
        self.ids = self.plant._mujoco_measurement_ids(model)
        self.pelvis_body = self.ids["pelvis_body"]
        self.plate_geom = self.ids["force_plate_geom"]
        self.ndof = int(model.nq)
        self.system_mass_kg = float(np.sum(np.asarray(model.body_mass[1:], dtype=np.float64)))
        joint_type_names = {
            int(mujoco.mjtJoint.mjJNT_FREE): "free",
            int(mujoco.mjtJoint.mjJNT_BALL): "ball",
            int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
            int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
        }
        for joint_name in REQUESTED_JOINT_AUDIT:
            jid = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_JOINT), joint_name)
            if jid < 0:
                self.joint_index_map[joint_name] = {
                    "available": False,
                    "qpos_index": None,
                    "qvel_index": None,
                    "joint_type": None,
                }
                continue
            joint_type = int(model.jnt_type[jid])
            qpos_index = int(model.jnt_qposadr[jid])
            qvel_index = int(model.jnt_dofadr[jid])
            # A free joint contributes quaternion translation coordinates to
            # qpos/qvel. The panel's pelvis entry is specifically the vertical
            # translation, so address the third coordinate explicitly.
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                qpos_index += 2
                qvel_index += 2
            self.joint_index_map[joint_name] = {
                "available": True,
                "joint_id": int(jid),
                "qpos_index": qpos_index,
                "qvel_index": qvel_index,
                "joint_type": joint_type_names.get(joint_type, str(joint_type)),
            }
        # Resolve exact landmark site ids for the trace layer.
        for site_name, _label in JOINT_TRACE_SITES:
            sid = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_SITE), site_name)
            if sid >= 0:
                self.joint_site_ids[site_name] = sid
                self.joint_hist[site_name] = []
        for site_name in FOOT_TRACE_SITES:
            sid = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_SITE), site_name)
            if sid >= 0:
                self.foot_site_ids[site_name] = sid
        self.bar_lpt_site_id = self.ids["bar_lpt_site"]
        for site_name in (
            "pelvis_site", "sternum_site", "head_site",
            "left_bar_grip_site", "right_bar_grip_site",
            "left_hip_site", "right_hip_site", "left_knee_site", "right_knee_site",
            "left_ankle_site", "right_ankle_site",
            "left_heel_site", "right_heel_site", "left_toe_site", "right_toe_site",
        ):
            sid = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_SITE), site_name)
            if sid >= 0:
                self.soft_tissue_site_ids[site_name] = int(sid)
        for body_name in ("left_upper_arm", "right_upper_arm"):
            bid = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_BODY), body_name)
            if bid >= 0:
                self.soft_tissue_body_ids[body_name] = int(bid)
        # Render-only: the single rigged skin is the visible subject, so the
        # plant's own humanoid/contact geoms are transparent in memory only.
        for gname in GEOM_HIDE_HUMANOID:
            gid = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_GEOM), gname)
            if gid >= 0:
                model.geom_rgba[gid] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        bar_gid = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_GEOM), "bar_capsule")
        if bar_gid >= 0:
            model.geom_rgba[bar_gid] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        plate_gid = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_GEOM), "force_plate")
        if plate_gid >= 0:
            model.geom_rgba[plate_gid] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Exact anatomical landmark site ids (foreground markers + graph rows).
        for site_name, _label in LANDMARK_SITES:
            sid = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_SITE), site_name)
            if sid >= 0:
                self.landmark_site_ids[site_name] = sid
                self.landmark_hist[site_name] = []
        self.bar_body_id = self.ids["bar_body"]
        # Static LPT base height anchored to exact force-plate geometry; the first
        # rendered quiet-stance frame fixes x/y once, then only the tether endpoint
        # follows the live bar attachment.
        plate_pos = np.array(model.geom_pos[self.plate_geom], dtype=np.float64)
        plate_size = np.array(model.geom_size[self.plate_geom], dtype=np.float64)
        self.plate_top_z = float(plate_pos[2] + plate_size[2])
        self.plate_half_y = float(plate_size[1])
        self.lpt_device_pos = np.array(
            [plate_pos[0] - 0.52, -(plate_size[1] + 0.18), self.plate_top_z + 0.001],
            dtype=np.float64,
        )

    def hook(self, original_step):
        def stepping_hook(m, d, nstep=1, *args, **kwargs):
            original_step(m, d, nstep, *args, **kwargs)
            self.step_count += 1
            if self.step_count <= self.settle_steps:
                return
            self._ensure_renderer(m)
            trial_index = self.step_count - self.settle_steps - 1
            state_pos = np.array(d.qpos, dtype=np.float64)
            state_vel = np.array(d.qvel, dtype=np.float64)
            self.pos_samples.append(state_pos)
            self.vel_samples.append(state_vel)
            self.root_z_world.append(float(d.xpos[self.pelvis_body][2]))
            com = np.array(d.subtree_com[self.pelvis_body], dtype=np.float64)
            self.com_points.append(com)
            if self.lpt_quiet_com_x is None:
                self.lpt_quiet_com_x = float(com[0])
            # Exact-state landmark histories (read-only; no authoring).
            for site_name, sid in self.joint_site_ids.items():
                self.joint_hist[site_name].append(np.array(d.site_xpos[sid], dtype=np.float64))
            for site_name, sid in self.landmark_site_ids.items():
                self.landmark_hist[site_name].append(np.array(d.site_xpos[sid], dtype=np.float64))
            self.bar_lpt_points.append(np.array(d.site_xpos[self.bar_lpt_site_id], dtype=np.float64))
            if trial_index in self.sample_step_indices:
                self._render_frame(m, d, trial_index)
        return stepping_hook

    def _foot_contact_state(self, model: Any, data: Any):
        left, right, per_fz, _ = self.plant._contact_forces(model, data, self.ids)
        thr = float(self.plant.CONTACT_FORCE_THRESHOLD_N)
        contacts = {g: per_fz[g] > thr for g in self.plant.FOOT_CONTACT_GEOMS}
        return left + right, contacts

    def _render_frame(self, model: Any, data: Any, trial_index: int) -> None:
        total_fz, contacts = self._foot_contact_state(model, data)
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.scene_option)
        scene = self.renderer.scene
        try:
            cam_y = float(scene.camera[0].pos[1])
        except Exception:
            cam_y = -10.0
        foreground_sign = 1.0 if cam_y >= 0.0 else -1.0
        _add_forceplate_zone(scene, model, data, self.plate_geom)
        use_makehuman_mesh = bool(self.makehuman is not None and self.makehuman.available)
        if not use_makehuman_mesh:
            # No MakeHuman asset at all: procedural capsule stand-in (disabled path).
            _add_rigged_makehuman_skin(scene, data, self.soft_tissue_site_ids, self.soft_tissue_body_ids)
        elif not self.human_is_skin:
            # Segmented-OBJ fallback: it needs procedural seam caps + limb overlays
            # to close chunk gaps. The true skin path below needs none of these.
            _add_makehuman_mesh_seam_caps(scene, data, self.soft_tissue_site_ids, self.soft_tissue_body_ids)
            _add_clean_bar_limb_overlays(scene, data, self.soft_tissue_site_ids)
        # Instrumented force-plate sensor zones: faint tiles on the plate top under
        # each exact foot landmark, brighter when that region is loaded.
        for site_name, sid in self.foot_site_ids.items():
            fp = np.array(data.site_xpos[sid], dtype=np.float64)
            loaded = contacts.get(site_name[:-5], False)
            tile = (0.32, 0.72, 0.46, 0.42) if loaded else (0.34, 0.40, 0.50, 0.24)
            _add_plate_tile(scene, float(fp[0]), float(fp[1]), self.plate_top_z, tile)
        # Fixed vertical ruler (0.1 m ticks), tall enough for the head and apex.
        _add_vertical_ruler(scene, x=0.62, y=0.0, z0=0.0, z1=1.90)
        # Loaded barbell racked on the exact bar body (steel sleeves, collars, plates).
        # No synthetic soft-tissue shelf: the bar is held by the retargeted arms/hands on
        # the exact grip sites and rests against the real retargeted upper-trunk/shoulder
        # MakeHuman skin (measured 4-7 px gap, no intersection); no fake support geometry.
        _add_barbell(scene, data, self.bar_body_id)
        # Foot ground-contact points: the three modeled contact patches (heel / forefoot /
        # toe) per foot, from the exact contact-geom sites, pulled to the camera-near sole
        # edge so they stay visible past the foot. Bright green when that region carries
        # force, dim otherwise -> shows the heel->forefoot->toe rollover of the segmented foot.
        for g in self.plant.FOOT_CONTACT_GEOMS:
            sid = self.foot_site_ids.get(f"{g}_site")
            if sid is None:
                continue
            pos = np.array(data.site_xpos[sid], dtype=np.float64)
            pos[1] += foreground_sign * 0.045
            # Render marker center above the deck; contact truth still comes
            # from the exact MuJoCo foot geom contact state.
            pos[2] = max(float(pos[2]) + 0.008, self.plate_top_z + 0.018)
            lit = bool(contacts.get(g, False))
            _add_sphere(scene, pos, 0.017 if lit else 0.011,
                        (0.34, 0.95, 0.48, 1.0) if lit else (0.62, 0.40, 0.40, 0.92),
                        emission=0.65 if lit else 0.12)
        # Bar / LPT attachment: recent cyan measurement trace + marker (distinct from COM yellow).
        bar_now = self.bar_lpt_points[-1]
        _add_trail(
            scene,
            self.bar_lpt_points,
            COL_BAR_TRACE,
            radius=0.007,
            max_points=120,
            target_segments=48,
        )
        _add_sphere(scene, bar_now, 0.026, COL_BAR_TRACE, emission=0.5)
        com_now = self.com_points[-1]
        # The SINGLE ground LPT device: fixed floor spot IN FRONT of the plate
        # (camera-near side), x-aligned to the exact first quiet-stance COM
        # vertical axis. The base does not chase COM; only the tether endpoint
        # follows the live bar site, preserving bar-only LPT truth.
        lpt_x = self.lpt_quiet_com_x if self.lpt_quiet_com_x is not None else float(com_now[0])
        self.lpt_device_pos = np.array(
            [
                float(lpt_x),
                foreground_sign * (FORCE_PLATE_VISUAL_HALF[1] + 0.28),
                self.plate_top_z + 0.001,
            ],
            dtype=np.float64,
        )
        lpt_spool = _add_lpt_device(scene, self.lpt_device_pos, bar_now, foreground_sign)
        # Context-only faint 3D COM path at exact system COM (body + bar). The
        # foreground-visible COM marker/trail is drawn later in screen space from
        # these same exact points after the render + composite pass.
        _add_com_path(scene, list(self.com_points[-COM_SCREEN_TRAIL_STEPS:]), (0.98, 0.83, 0.30, 0.42))
        # Ground-reaction arrow from the plate up, length scaled by total Fz.
        if total_fz > float(self.plant.CONTACT_FORCE_THRESHOLD_N):
            base = np.array(data.geom_xpos[self.plate_geom], dtype=np.float64)
            base[0] = float(com_now[0]); base[2] = self.plate_top_z + 0.03
            length = _bound(total_fz / 9000.0, 0.05, 0.9)
            tip = base.copy(); tip[2] = base[2] + length
            _add_segment(scene, base, tip, 0.018, (0.47, 0.82, 0.59, 0.95),
                         kind=int(mujoco.mjtGeom.mjGEOM_ARROW))

        # --- Truthful instrumented 3D pass, then mesh-only athlete composite. --- #
        base_rgb = np.ascontiguousarray(self.renderer.render()[:, :, :3])
        self.renderer.enable_depth_rendering()
        base_depth = np.asarray(self.renderer.render(), dtype=np.float64)
        self.renderer.disable_depth_rendering()
        rgb = base_rgb
        human_used = not use_makehuman_mesh
        if use_makehuman_mesh and self.makehuman is not None:
            rgb, human_used = self.makehuman.composite(base_rgb, base_depth, data)
        self.human_visual_used = self.human_visual_used or bool(human_used)

        # --- Foreground screen-space geometry (always visible over the human) ---- #
        trail_points = self.com_points[-COM_SCREEN_TRAIL_STEPS:]
        step = max(1, len(trail_points) // 90)
        com_screen_path = [
            xy for xy in (
                _project_world_to_screen(scene, np.asarray(p, dtype=np.float64), self.width, self.height)
                for p in trail_points[::step]
            )
            if xy is not None
        ]
        current_xy = _project_world_to_screen(scene, com_now, self.width, self.height)
        if current_xy is not None and (not com_screen_path or com_screen_path[-1] != current_xy):
            com_screen_path.append(current_xy)
        landmark_screen: list[dict[str, Any]] = []
        landmark_labels = dict(LANDMARK_SITES)
        for site_name, sid in self.landmark_site_ids.items():
            xy = _project_world_to_screen(
                scene, np.asarray(data.site_xpos[sid], dtype=np.float64), self.width, self.height)
            if xy is None:
                continue
            trace: list[tuple[int, int]] = []
            if site_name in LANDMARK_TRACE_SITES:
                pts = self.landmark_hist.get(site_name, [])[-JOINT_TRAIL_STEPS:]
                step_lm = max(1, len(pts) // 42)
                trace = [
                    pxy for pxy in (
                        _project_world_to_screen(scene, np.asarray(p, dtype=np.float64),
                                                 self.width, self.height)
                        for p in pts[::step_lm]
                    )
                    if pxy is not None
                ]
            landmark_screen.append({
                "name": site_name,
                "label": landmark_labels.get(site_name, site_name),
                "xy": xy,
                "trace": trace,
            })
        # Single clean FORCE PLATE label anchored to the near-front edge of the deck.
        plate_label_pos = np.array(data.geom_xpos[self.plate_geom], dtype=np.float64)
        plate_label_pos[0] -= 0.70 * FORCE_PLATE_VISUAL_HALF[0]
        plate_label_pos[1] = -foreground_sign * (0.55 * FORCE_PLATE_VISUAL_HALF[1])
        plate_label_pos[2] = self.plate_top_z + 0.04
        plate_screen_xy = _project_world_to_screen(scene, plate_label_pos, self.width, self.height)
        lpt_screen_xy = None
        if lpt_spool is not None:
            lpt_screen_xy = _project_world_to_screen(
                scene, np.asarray(lpt_spool, dtype=np.float64), self.width, self.height)
        barbell_screen_xy = _project_world_to_screen(
            scene, np.asarray(data.xpos[self.bar_body_id], dtype=np.float64), self.width, self.height)
        self.frames.append({
            "trial_index": trial_index,
            "rgb": np.ascontiguousarray(rgb[:, :, :3]),
            "com_screen_path": com_screen_path,
            "landmark_screen": landmark_screen,
            "force_plate_screen_xy": plate_screen_xy,
            "lpt_screen_xy": lpt_screen_xy,
            "barbell_screen_xy": barbell_screen_xy,
            "makehuman_visual_used": human_used,
        })


# --------------------------------------------------------------------------- #
# 2D overlay composition (telemetry panel + phase strip + event flash)        #
# --------------------------------------------------------------------------- #
def _fmt(value: float, nd: int = 3) -> str:
    return f"{value:.{nd}f}"


def _window_indices(time_s: list[float], window: tuple[float, float]) -> list[int]:
    t0, t1 = window
    return [i for i, t in enumerate(time_s) if t0 <= t <= t1]


def _plot_series(img: np.ndarray, *, time_s: list[float], values: list[float],
                 window: tuple[float, float], rect: tuple[int, int, int, int],
                 y_min: float, y_max: float, color: tuple[int, int, int],
                 t: int = 1, alpha: float = 1.0, stride: int = 1) -> None:
    x0, y0, x1, y1 = rect
    t0, t1 = window
    span_t = max(t1 - t0, 1e-9)
    span_y = max(y_max - y_min, 1e-9)
    points: list[tuple[int, int]] = []
    for i in range(0, min(len(time_s), len(values)), max(1, stride)):
        tt = float(time_s[i])
        if tt < t0 or tt > t1:
            continue
        yy = _bound((float(values[i]) - y_min) / span_y, 0.0, 1.0)
        px = int(x0 + (x1 - x0) * ((tt - t0) / span_t))
        py = int(y1 - (y1 - y0) * yy)
        points.append((px, py))
    if len(points) >= 2:
        draw_polyline(img, points, color, t=t, alpha=alpha)


def _draw_com_foreground(
    img: np.ndarray,
    screen_path: list[tuple[int, int]] | None,
    placed: list[tuple[int, int, int, int]] | None = None,
) -> None:
    if not screen_path:
        return
    trail = screen_path[-90:]
    for i in range(1, len(trail)):
        frac = i / max(len(trail) - 1, 1)
        col = (
            int(130 + 125 * frac),
            int(106 + 118 * frac),
            int(32 + 42 * frac),
        )
        draw_line(img, trail[i - 1][0], trail[i - 1][1], trail[i][0], trail[i][1],
                  col, t=2, alpha=0.20 + 0.62 * frac)
    cx, cy = trail[-1]
    draw_circle(img, cx, cy, 14, (6, 8, 12), alpha=0.88)
    draw_circle(img, cx, cy, 11, (255, 255, 245), alpha=0.96)
    draw_circle(img, cx, cy, 8, (255, 220, 72), alpha=1.0)
    draw_circle(img, cx, cy, 4, (8, 10, 14), alpha=0.90)
    draw_circle(img, cx, cy, 2, (255, 244, 120), alpha=1.0)
    if placed is not None:
        _draw_callout_label(
            img, (cx, cy), "SYSTEM COM", COL_COM, placed,
            candidates=((18, -24), (18, 14), (-96, -24), (-96, 14)),
        )


def _intersects_any(rect: tuple[int, int, int, int],
                    placed: list[tuple[int, int, int, int]]) -> bool:
    x0, y0, x1, y1 = rect
    for ax0, ay0, ax1, ay1 in placed:
        if x0 < ax1 and x1 > ax0 and y0 < ay1 and y1 > ay0:
            return True
    return False


def _draw_callout_label(
    img: np.ndarray,
    anchor: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
    placed: list[tuple[int, int, int, int]],
    *,
    candidates: tuple[tuple[int, int], ...] = (
        (12, -12), (12, 8), (-76, -12), (-76, 8),
        (18, -28), (-86, -28), (18, 20), (-86, 20),
    ),
) -> tuple[int, int, int, int]:
    h, w = img.shape[:2]
    ax, ay = anchor
    tw = _text_width(label, 1, 1)
    chosen: tuple[int, int, int, int] | None = None
    for dx, dy in candidates:
        x0 = max(2, min(w - tw - 12, ax + dx))
        y0 = max(48, min(h - 18, ay + dy))
        rect = (x0, y0, x0 + tw + 10, y0 + 14)
        padded = (rect[0] - 4, rect[1] - 4, rect[2] + 4, rect[3] + 4)
        if not _intersects_any(padded, placed):
            chosen = rect
            break
    if chosen is None:
        x0 = max(2, min(w - tw - 12, ax + candidates[0][0]))
        y0 = max(48, min(h - 18, ay + candidates[0][1]))
        chosen = (x0, y0, x0 + tw + 10, y0 + 14)
    x0, y0, x1, y1 = chosen
    draw_line(img, ax, ay, x0 if x0 > ax else x1, (y0 + y1) // 2, color, t=1, alpha=0.62)
    fill_rect(img, x0, y0, x1, y1, (7, 9, 12), 0.76)
    rect_outline(img, x0, y0, x1, y1, color, 1)
    draw_text(img, x0 + 5, y0 + 3, label, color, scale=1)
    placed.append((x0 - 4, y0 - 4, x1 + 4, y1 + 4))
    return chosen


def _right_dashboard_bounds(width: int) -> tuple[int, int, int, int]:
    """Shared right-column geometry: panels align, meter owns the far-right edge."""
    meter_x1 = width - RIGHT_METER_RIGHT_LABEL_GUTTER_PX
    meter_x0 = meter_x1 - RIGHT_METER_WIDTH_PX
    panel_x0 = width - RIGHT_DASHBOARD_WIDTH_PX
    panel_x1 = meter_x0 - RIGHT_METER_LABEL_GUTTER_PX
    return panel_x0, panel_x1, meter_x0, meter_x1


def _right_time_plot_bounds(width: int) -> tuple[int, int]:
    panel_x0, panel_x1, _, _ = _right_dashboard_bounds(width)
    return panel_x0 + RIGHT_TIME_PLOT_LEFT_INSET_PX, panel_x1


def _draw_force_time_graph(
    img: np.ndarray,
    *,
    t_s: float,
    window: tuple[float, float],
    phase_spans: list[tuple[float, float, str]],
    graph: dict[str, Any],
) -> int:
    h, w = img.shape[:2]
    gx0, gx1 = _right_time_plot_bounds(w)
    gy0, gy1 = RIGHT_FZ_GRAPH_Y0, h - RIGHT_FZ_GRAPH_BOTTOM_MARGIN_PX
    t0, t1 = window
    span = max(t1 - t0, 1e-9)

    def _to_x(t: float) -> int:
        return int(gx0 + (gx1 - gx0) * _bound((t - t0) / span, 0.0, 1.0))

    fill_rect(img, gx0, gy0 - 58, gx1, gy1 + 26, COL_BG_PANEL, 0.72)
    rect_outline(img, gx0, gy0 - 58, gx1, gy1 + 26, (60, 68, 80), 1)
    for a, b, name in phase_spans:
        xa, xb = _to_x(a), _to_x(b)
        if xb > xa:
            fill_rect(img, xa, gy0, xb, gy1, PHASE_COLORS_255.get(name, COL_DIM), 0.14)
            phase_label = PHASE_LABELS.get(name, name)
            label_w = _text_width(phase_label, 1, 1)
            if xb - xa > label_w + 8:
                draw_text(img, xa + (xb - xa - label_w) // 2, gy0 + 4,
                          phase_label, PHASE_COLORS_255.get(name, COL_DIM), scale=1)
            elif xb - xa > 22:
                abbr = PHASE_ABBR.get(name, phase_label[:3])
                draw_text(img, xa + 2, gy0 + 4, abbr,
                          PHASE_COLORS_255.get(name, COL_DIM), scale=1)
    draw_text(img, gx0, gy0 - 50, "FORCE-PLATE FZ-TIME GRAPH (N)", COL_TEXT, scale=1)
    draw_text(img, gx0, gy0 - 36, "TOTAL FZ", COL_FORCE, scale=1)
    draw_text(img, gx0 + 74, gy0 - 36, "LEFT FZ", (236, 184, 96), scale=1)
    draw_text(img, gx0 + 136, gy0 - 36, "RIGHT FZ", (128, 184, 238), scale=1)
    if graph.get("com_inertial_fz_N") is not None:
        draw_text(img, gx0 + 216, gy0 - 36, "COM INERTIAL FZ EST", COL_ACCENT, scale=1)
    draw_text(img, gx0, gy0 - 22, "YELLOW=BW+LOAD  WHITE=FRAME CURSOR", (236, 224, 112), scale=1)

    time_s = graph["time_s"]
    fz_total = graph["fz_total_N"]
    fz_left = graph.get("fz_left_N")
    fz_right = graph.get("fz_right_N")
    ref_N = float(graph["bodyweight_N"])
    idx = _window_indices(time_s, window)
    peak_values = [float(fz_total[i]) for i in idx]
    if fz_left is not None:
        peak_values.extend(float(fz_left[i]) for i in idx)
    if fz_right is not None:
        peak_values.extend(float(fz_right[i]) for i in idx)
    fz_peak = max(peak_values or [ref_N])
    y_min, y_max = 0.0, max(ref_N * 3.0, fz_peak * 1.06, 1.0)
    plot_rect = (gx0, gy0, gx1, gy1)

    for frac in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0):
        yy = int(gy1 - (gy1 - gy0) * frac)
        fill_rect(img, gx0, yy, gx1, yy + 1, (42, 48, 58), 0.72)
    bw_y = int(gy1 - (gy1 - gy0) * _bound(ref_N / y_max, 0.0, 1.0))
    draw_line(img, gx0, bw_y, gx1, bw_y, (236, 224, 112), t=1, alpha=0.92)
    bw_label = "BW+LOAD"
    bw_label_x = gx0 + (gx1 - gx0 - _text_width(bw_label, 1, 1)) // 2
    fill_rect(
        img, bw_label_x - 3, bw_y - 11,
        bw_label_x + _text_width(bw_label, 1, 1) + 3, bw_y - 1,
        COL_BG_PANEL, 0.78,
    )
    draw_text(img, bw_label_x, bw_y - 9, bw_label, (236, 224, 112), scale=1)

    stride = max(1, len(time_s) // 720)
    if fz_left is not None:
        _plot_series(img, time_s=time_s, values=fz_left, window=window, rect=plot_rect,
                     y_min=y_min, y_max=y_max, color=(236, 184, 96), t=1, alpha=0.90, stride=stride)
    if fz_right is not None:
        _plot_series(img, time_s=time_s, values=fz_right, window=window, rect=plot_rect,
                     y_min=y_min, y_max=y_max, color=(128, 184, 238), t=1, alpha=0.90, stride=stride)
    _plot_series(img, time_s=time_s, values=fz_total, window=window, rect=plot_rect,
                 y_min=y_min, y_max=y_max, color=COL_FORCE, t=3, alpha=0.98, stride=stride)
    com_inertial = graph.get("com_inertial_fz_N")
    if com_inertial is not None:
        _plot_series(img, time_s=time_s, values=com_inertial, window=window, rect=plot_rect,
                     y_min=y_min, y_max=y_max, color=COL_ACCENT, t=1, alpha=0.62, stride=stride)
    rect_outline(img, gx0, gy0, gx1, gy1, (84, 92, 106), 1)

    cx = _to_x(t_s)
    draw_line(img, cx, gy0 - 2, cx, gy1 + 16, COL_TEXT, t=2, alpha=0.95)
    sample_i = int(graph.get("current_index", 0))
    sample_i = max(0, min(sample_i, len(fz_total) - 1))
    cursor_y = int(gy1 - (gy1 - gy0) * _bound(float(fz_total[sample_i]) / y_max, 0.0, 1.0))
    draw_circle(img, cx, cursor_y, 5, COL_FORCE, alpha=1.0)
    label_x = min(max(cx + 8, gx0 + 8), gx1 - 116)
    draw_text(img, label_x, max(gy0 + 4, cursor_y - 16),
              f"{_fmt(float(fz_total[sample_i]), 0)} N", COL_FORCE, scale=1)
    if fz_left is not None and fz_right is not None:
        draw_text(img, gx0 + 2, gy1 - 14,
                  f"L {_fmt(float(fz_left[sample_i]), 0)}  R {_fmt(float(fz_right[sample_i]), 0)} N",
                  COL_TEXT, scale=1)
    draw_text(img, gx0, gy1 + 7, "TIME (S)", COL_DIM, scale=1)
    draw_text(img, gx0 + 80, gy1 + 7, "PHASE SHADING = JUMP PHASE TIMELINE", COL_DIM, scale=1)
    return cx


def _draw_joint_panel(
    img: np.ndarray,
    *,
    panel: dict[str, Any] | None,
    current_index: int,
    window: tuple[float, float],
) -> None:
    if not panel or not panel.get("items"):
        return
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (
        LEFT_JOINT_PANEL_X0,
        LEFT_JOINT_PANEL_Y0,
        LEFT_JOINT_PANEL_X1,
        h - LEFT_JOINT_PANEL_BOTTOM_MARGIN_PX,
    )
    fill_rect(img, x0, y0, x1, y1, COL_BG_PANEL, 0.64)
    rect_outline(img, x0, y0, x1, y1, (60, 68, 80), 1)
    draw_text(img, x0 + 10, y0 + 8, "JOINT KINEMATICS", COL_DIM, scale=1)
    draw_text(img, x0 + 10, y0 + 20, "YELLOW = POSITION Q (M/RAD)", (235, 212, 104), scale=1)
    draw_text(img, x0 + 10, y0 + 32, "BLUE = VELOCITY QDOT (M/S OR RAD/S)", (108, 194, 232), scale=1)
    draw_line(img, x0 + 10, y0 + 42, x0 + 28, y0 + 42, (235, 212, 104), t=2)
    draw_line(img, x0 + 250, y0 + 42, x0 + 268, y0 + 42, (108, 194, 232), t=2)

    items = panel["items"]
    time_s = panel["time_s"]
    t0, t1 = window
    span = max(t1 - t0, 1e-9)
    row_h = 38
    plot_x0, plot_x1 = x0 + 220, x1 - 12
    start_y = y0 + 92
    cursor_t = float(time_s[max(0, min(current_index, len(time_s) - 1))])
    cursor_x = int(plot_x0 + (plot_x1 - plot_x0) * _bound((cursor_t - t0) / span, 0.0, 1.0))

    for row, item in enumerate(items[:9]):
        ry0 = start_y + row * row_h
        ry1 = ry0 + 26
        if ry1 > y1 - 8:
            break
        label = item["label"]
        series = item.get("series") or [{"side": "", "q": item["q"], "qdot": item["qdot"]}]
        q_now_parts = []
        for s in series:
            vals = s["q"]
            side = s.get("side", "")
            now = float(vals[max(0, min(current_index, len(vals) - 1))])
            q_now_parts.append(f"{side}{_fmt(now, 2)}" if side else _fmt(now, 2))
        draw_text(img, x0 + 10, ry0 + 5, label, COL_TEXT, scale=1)
        fill_rect(img, plot_x0, ry0, plot_x1, ry1, (28, 34, 44), 0.78)
        fill_rect(img, plot_x0, (ry0 + ry1) // 2, plot_x1, (ry0 + ry1) // 2 + 1, (78, 84, 96), 0.65)
        rect_outline(img, plot_x0, ry0, plot_x1, ry1, (58, 64, 76), 1)
        q_vals = [float(x) for s in series for x in s["q"]]
        qd_vals = [float(x) for s in series for x in s["qdot"]]
        q_span = max(max(abs(v) for v in q_vals), 1e-6)
        qd_span = max(max(abs(v) for v in qd_vals), 1e-6)
        side_cols = (
            ((235, 212, 104), (108, 194, 232)),
            ((250, 164, 84), (126, 224, 238)),
        )
        for sidx, s in enumerate(series[:2]):
            q_scaled = [0.5 + 0.44 * _bound(float(v) / q_span, -1.0, 1.0) for v in s["q"]]
            qd_scaled = [0.5 + 0.44 * _bound(float(v) / qd_span, -1.0, 1.0) for v in s["qdot"]]
            q_col, qd_col = side_cols[min(sidx, len(side_cols) - 1)]
            _plot_series(img, time_s=time_s, values=q_scaled, window=window,
                         rect=(plot_x0, ry0 + 1, plot_x1, ry1 - 1),
                         y_min=0.0, y_max=1.0, color=q_col, t=1, alpha=0.95,
                         stride=max(1, len(time_s) // 400))
            _plot_series(img, time_s=time_s, values=qd_scaled, window=window,
                         rect=(plot_x0, ry0 + 1, plot_x1, ry1 - 1),
                         y_min=0.0, y_max=1.0, color=qd_col, t=1, alpha=0.70,
                         stride=max(1, len(time_s) // 400))
        draw_line(img, cursor_x, ry0 - 1, cursor_x, ry1 + 1, COL_TEXT, t=1, alpha=0.86)
        q_readout = " / ".join(q_now_parts)
        draw_text(img, x0 + 132, ry0 + 5, q_readout, COL_DIM, scale=1)


def _central_second_derivative(time_s: list[float], values: list[float]) -> list[float]:
    n = min(len(time_s), len(values))
    if n < 3:
        return [0.0] * n
    acc = [0.0] * n
    for i in range(1, n - 1):
        h0 = max(float(time_s[i]) - float(time_s[i - 1]), 1e-9)
        h1 = max(float(time_s[i + 1]) - float(time_s[i]), 1e-9)
        s0 = (float(values[i]) - float(values[i - 1])) / h0
        s1 = (float(values[i + 1]) - float(values[i])) / h1
        acc[i] = 2.0 * (s1 - s0) / (h0 + h1)
    acc[0] = acc[1]
    acc[-1] = acc[-2]
    return acc


def _compose_overlay(frame: np.ndarray, *, t_s: float, phase: str, fz_N: float,
                     root_z: float, com_z: float, bar_z: float, bar_vel: float,
                     bar_disp: float, bodyweight_N: float,
                     window: tuple[float, float], phase_spans: list[tuple[float, float, str]],
                     events: dict[str, float], slowmo: float, event_flash: str | None,
                     grf: tuple[float, float, float, str] | None = None,
                     grf_history: dict[str, Any] | None = None,
                     cop_x: float | None = None,
                     force_graph: dict[str, Any] | None = None,
                     joint_panel: dict[str, Any] | None = None,
                     current_index: int = 0,
                     com_screen_path: list[tuple[int, int]] | None = None,
                     force_plate_screen_xy: tuple[int, int] | None = None,
                     landmark_screen: list[dict[str, Any]] | None = None,
                     lpt_screen_xy: tuple[int, int] | None = None,
                     barbell_screen_xy: tuple[int, int] | None = None) -> None:
    h, w = frame.shape[:2]

    # Title band (top-left).
    fill_rect(frame, 0, 0, w, 46, COL_BG_PANEL, 0.55)
    draw_text(frame, 18, 10, "LOADED COUNTERMOVEMENT JUMP", COL_TEXT, scale=3, spacing=2)
    draw_text(frame, 18, 34, "AUTHORITATIVE ROLLOUT  MAKEHUMAN MESH  BODY 75 KG + BAR 20 KG", COL_DIM, scale=1, spacing=1)
    draw_text(frame, w - 250, 12, f"SLOW-MO {_fmt(slowmo, 2)}X", COL_ACCENT, scale=2, spacing=1)
    draw_text(frame, w - 250, 30, "REAL TIME SHOWN", COL_DIM, scale=1, spacing=1)

    # Force-plate GRF distribution panel (upper-right, under the title): heel/forefoot/
    # toe vertical-force shares, center of pressure, and the dominant loaded region
    # tracing the heel->forefoot->toe rollover. All read from the force-plate trace.
    if grf is not None:
        heel_s, fore_s, toe_s, dom = grf
        right_panel_x0, right_panel_x1, _, _ = _right_dashboard_bounds(w)
        gpx0, gpy0, gpx1, gpy1 = right_panel_x0, RIGHT_GRF_PANEL_Y0, right_panel_x1, RIGHT_GRF_PANEL_Y1
        fill_rect(frame, gpx0, gpy0, gpx1, gpy1, COL_BG_PANEL, 0.62)
        rect_outline(frame, gpx0, gpy0, gpx1, gpy1, (60, 68, 80), 1)
        draw_text(frame, gpx0 + 10, gpy0 + 8, "FORCE-PLATE GRF CONTACT SHARE", COL_DIM, scale=1)
        draw_text(frame, gpx0 + 10, gpy0 + 20, "HEEL / FOREFOOT / TOE FORCE-SHARE REGIONS", COL_DIM, scale=1)
        rollover_col = COL_EVENT if dom in ("HEEL", "FORE", "TOE") else COL_DIM
        dom_label = {"FORE": "FOREFOOT", "HEEL": "HEEL", "TOE": "TOE", "AIR": "AIR"}.get(dom, dom)
        draw_text(frame, gpx1 - 178, gpy0 + 8, f"LOADED REGION: {dom_label}", rollover_col, scale=1)
        bars = (("HEEL", heel_s, (216, 150, 92)), ("FORE", fore_s, (110, 200, 128)),
                ("TOE", toe_s, (120, 176, 232)))
        by = gpy0 + 38
        for label, share, col in bars:
            draw_text(frame, gpx0 + 10, by + 2, label, COL_TEXT, scale=1)
            bx0, bx1 = gpx0 + 66, gpx1 - 52
            fill_rect(frame, bx0, by, bx1, by + 14, (40, 46, 56), 0.9)
            fw = int((bx1 - bx0) * _bound(share, 0.0, 1.0))
            fill_rect(frame, bx0, by, bx0 + fw, by + 14, col, 0.95)
            draw_text(frame, bx1 + 6, by + 2, f"{int(round(share * 100)):3d}%", COL_TEXT, scale=1)
            by += 21
        cop_txt = "AIR" if cop_x is None else f"{cop_x:+.3f} M"
        draw_text(frame, gpx0 + 10, by + 2, f"CENTER OF PRESSURE (COP X) = {cop_txt}", COL_ACCENT, scale=1)
        if grf_history is not None:
            hx0, hx1 = _right_time_plot_bounds(w)
            hy0, hy1 = gpy1 - 44, gpy1 - 12
            fill_rect(frame, hx0, hy0, hx1, hy1, (24, 29, 38), 0.78)
            rect_outline(frame, hx0, hy0, hx1, hy1, (58, 64, 76), 1)
            draw_text(frame, gpx0 + 10, hy0 + 9, "TRACE", COL_DIM, scale=1)
            hist_time = grf_history["time_s"]
            stride = max(1, len(hist_time) // 360)
            for key, col in (("heel_share", (216, 150, 92)), ("fore_share", (110, 200, 128)),
                             ("toe_share", (120, 176, 232))):
                _plot_series(frame, time_s=hist_time, values=grf_history[key], window=window,
                             rect=(hx0, hy0 + 2, hx1, hy1 - 2), y_min=0.0, y_max=1.0,
                             color=col, t=1, alpha=0.95, stride=stride)
            t0, t1 = window
            cursor_x = int(hx0 + (hx1 - hx0) * _bound((t_s - t0) / max(t1 - t0, 1e-9), 0.0, 1.0))
            draw_line(frame, cursor_x, hy0 - 1, cursor_x, hy1 + 1, COL_TEXT, t=1, alpha=0.9)

    # Telemetry panel (bottom-left). Force plate is primary (FZ in N); the LPT is
    # secondary bar-only kinematics and leads with velocity (m/s) then displacement
    # (m); LPT force is only a tiny measurement-only note (SLICE_B_DASHBOARD_UNITS).
    px0, telem_h, px1, telem_bottom = LOWER_LEFT_TELEMETRY_BOUNDS
    py0, py1 = h - telem_h, h - telem_bottom
    fill_rect(frame, px0, py0, px1, py1, COL_BG_PANEL, 0.66)
    rect_outline(frame, px0, py0, px1, py1, (60, 68, 80), 1)
    tx, ty = px0 + 12, py0 + 10
    draw_text(frame, tx, ty, f"T = {_fmt(t_s)} S", COL_TEXT, scale=2)
    pc = PHASE_COLORS_255.get(phase, COL_TEXT)
    draw_text(frame, tx + 150, ty, PHASE_SHORT.get(phase, phase), pc, scale=2)
    draw_text(frame, tx, ty + 22, f"FZ = {_fmt(fz_N, 0)} N  (FORCE PLATE)", COL_FORCE, scale=1)
    draw_text(frame, tx, ty + 38, f"ROOT Z = {_fmt(root_z)} M", COL_TEXT, scale=2)
    draw_text(frame, tx, ty + 60, f"COM  Z = {_fmt(com_z)} M", COL_COM, scale=2)
    draw_text(frame, tx, ty + 84, f"LPT VELOCITY = {_fmt(bar_vel)} M/S", COL_BAR_TRACE_255, scale=2)
    draw_text(frame, tx, ty + 106, f"LPT DISPLACEMENT = {_fmt(bar_disp)} M", COL_BAR_TRACE_255, scale=2)
    draw_text(frame, tx, ty + 128, f"BAR Z = {_fmt(bar_z)} M   LPT IS BAR-ONLY", COL_DIM, scale=1)
    draw_text(frame, tx, ty + 142, LPT_TETHER_TELEMETRY_LABEL, COL_DIM, scale=1)

    # Far-right normalized Fz/BW gauge with a bodyweight reference tick.
    _, _, meter_x0, meter_x1 = _right_dashboard_bounds(w)
    gx0, gy0, gx1, gy1 = meter_x0, 54, meter_x1, h - 40
    fill_rect(frame, gx0 - 2, gy0 - 2, gx1 + 2, gy1 + 2, COL_BG_PANEL, 0.6)
    fz_ref = max(bodyweight_N * 3.0, 1.0)
    fh = int((gy1 - gy0) * _bound(fz_N / fz_ref, 0.0, 1.0))
    fill_rect(frame, gx0, gy1 - fh, gx1, gy1, COL_FORCE, 0.9)
    for ratio, col in ((0.0, COL_DIM), (1.0, (232, 228, 120)), (1.5, COL_DIM),
                       (2.0, COL_DIM), (3.0, COL_TEXT)):
        yy = int(gy1 - (gy1 - gy0) * _bound(ratio / 3.0, 0.0, 1.0))
        draw_line(frame, gx0 - 5, yy, gx1 + 5, yy, col, t=1, alpha=0.86)
        label = "0" if ratio == 0.0 else f"{ratio:.1f}"
        draw_text(frame, gx0 - 28, yy - 4, label, col, scale=1)
    current_y = int(gy1 - (gy1 - gy0) * _bound((fz_N / max(bodyweight_N, 1.0)) / 3.0, 0.0, 1.0))
    draw_line(frame, gx0 - 8, current_y, gx1 + 8, current_y, COL_EVENT, t=2, alpha=0.98)
    draw_text(frame, gx0 - 34, gy0 - 18, "FZ/BW", COL_DIM, scale=1)
    draw_text(frame, gx0 - 40, current_y - 14, _fmt(fz_N / max(bodyweight_N, 1.0), 2), COL_EVENT, scale=1)
    bw_y = int(gy1 - (gy1 - gy0) * _bound(1.0 / 3.0, 0.0, 1.0))
    bw_meter_label_x = min(gx1 + 2, w - _text_width("+LOAD", 1, 1) - 2)
    draw_text(frame, bw_meter_label_x, bw_y - 12, "BW", (232, 228, 120), scale=1)
    draw_text(frame, bw_meter_label_x, bw_y - 2, "+LOAD", (232, 228, 120), scale=1)

    # Compact force-time graph in the upper-right stack. Primary curve is the
    # measured force-plate Fz trace; optional COM inertial force is secondary.
    if force_graph is not None:
        force_graph["current_index"] = current_index
        _draw_force_time_graph(
            frame, t_s=t_s, window=window, phase_spans=phase_spans,
            graph=force_graph,
        )

    _draw_joint_panel(
        frame, panel=joint_panel, current_index=current_index, window=window,
    )

    # Phase strip timeline (bottom center) with event ticks and a live cursor.
    sx0, sx1 = 410, w - 126
    sy0, sy1 = h - 60, h - 40
    t0, t1 = window
    span = max(t1 - t0, 1e-6)

    def _to_x(t: float) -> int:
        return int(sx0 + (sx1 - sx0) * _bound((t - t0) / span, 0.0, 1.0))

    fill_rect(frame, sx0 - 2, sy0 - 18, sx1 + 2, sy1 + 16, COL_BG_PANEL, 0.6)
    draw_text(frame, sx0, sy0 - 14, "JUMP PHASE TIMELINE", COL_DIM, scale=1)
    draw_text(frame, sx1 - 120, sy0 - 14, "CURRENT TIME CURSOR", COL_TEXT, scale=1)
    for a, b, name in phase_spans:
        xa, xb = _to_x(a), _to_x(b)
        if xb <= xa:
            continue
        fill_rect(frame, xa, sy0, xb, sy1, PHASE_COLORS_255.get(name, COL_DIM), 0.92)
        phase_label = PHASE_LABELS.get(name, name)
        label_w = _text_width(phase_label, 1, 1)
        if xb - xa > label_w + 6:
            draw_text(frame, xa + (xb - xa - label_w) // 2, sy0 + 6,
                      phase_label, COL_TEXT, scale=1)
        elif xb - xa > 20:
            abbr = PHASE_ABBR.get(name, phase_label[:3])
            draw_text(frame, xa + 2, sy0 + 6, abbr, COL_TEXT, scale=1)
    for label, key in (("TO", "takeoff_time_s"), ("APEX", "apex_time_s"), ("LAND", "landing_time_s")):
        tv = events.get(key)
        if tv is None:
            continue
        ex = _to_x(float(tv))
        fill_rect(frame, ex - 1, sy0 - 10, ex + 1, sy1 + 2, COL_EVENT)
        draw_text(frame, ex - 10, sy0 - 22, label, COL_EVENT, scale=1)
    cx = _to_x(t_s)
    fill_rect(frame, cx - 1, sy0 - 6, cx + 1, sy1 + 6, COL_TEXT)
    # Anatomical joint landmarks + short fading traces, projected from EXACT CMJ
    # sites. These on-body kinematic landmarks mirror the JOINT KINEMATICS panel
    # rows on the left (same joint names). Drawn before COM so COM stays on top.
    placed_labels: list[tuple[int, int, int, int]] = []
    if landmark_screen:
        for lm in landmark_screen:
            trace = lm.get("trace") or []
            for i in range(1, len(trace)):
                frac = i / max(len(trace) - 1, 1)
                draw_line(frame, trace[i - 1][0], trace[i - 1][1], trace[i][0], trace[i][1],
                          JOINT_LANDMARK_RED, t=2, alpha=0.035 + 0.16 * frac)
        for lm in landmark_screen:
            px, py = lm["xy"]
            draw_circle(frame, px, py, 7, JOINT_LANDMARK_OUTLINE, alpha=0.92)
            draw_circle(frame, px, py, 5, JOINT_LANDMARK_RED, alpha=1.0)
            draw_circle(frame, px, py, 2, (255, 210, 210), alpha=0.90)

    # Post-render, foreground-visible COM overlay. The 2D points are projected
    # from exact MuJoCo COM samples captured during the authoritative rollout.
    _draw_com_foreground(frame, com_screen_path, placed_labels)

    # Screen-space hardware labels, projected from exact scene positions. A single
    # FORCE PLATE label (the left/right Fz split is shown in the graph + telemetry,
    # which is unambiguous, unlike two coincident zone labels in the sagittal view).
    if force_plate_screen_xy is not None:
        _draw_callout_label(
            frame, force_plate_screen_xy, "FORCE PLATES", COL_TEXT, placed_labels,
            candidates=((-150, -48), (-178, -48), (-150, -30), (-178, -30)),
        )
    if lpt_screen_xy is not None:
        _draw_callout_label(
            frame, lpt_screen_xy, LPT_SCENE_LABEL, COL_BAR_TRACE_255, placed_labels,
            candidates=((44, -8), (44, -22), (-84, -8), (-84, -22), (58, -36), (-98, -36)),
        )
    if event_flash:
        ew = _text_width(event_flash, 2, 1)
        ex = (w - ew) // 2
        fill_rect(frame, ex - 12, 50, ex + ew + 12, 82, (20, 16, 12), 0.62)
        draw_text(frame, ex, 58, event_flash, COL_EVENT, scale=2, spacing=1)


# --------------------------------------------------------------------------- #
# Video encode via ffmpeg raw pipe                                            #
# --------------------------------------------------------------------------- #
def _open_ffmpeg(ffmpeg: str, output: Path, width: int, height: int, fps: int):
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-movflags", "+faststart", str(output),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a loaded CMJ rollout with scientific overlays.")
    parser.add_argument("--output-dir", type=Path, default=Path("media"))
    parser.add_argument("--params", type=Path, help="JSON parameter configuration")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--duration-sec", type=float)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--telemetry-json", type=Path)
    args = parser.parse_args(argv)

    if args.width <= 0 or args.height <= 0:
        raise ValueError("render width and height must be positive")
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    script_dir = Path(__file__).resolve().parent
    repository_root = script_dir.parents[1]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write rendering.mp4")

    plant = _load_public_plant(repository_root / "src" / "loaded_cmj" / "plant.py")
    params_path = args.params.resolve() if args.params is not None else output_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"parameter configuration does not exist: {params_path}"
        )
    params = plant.load_params(params_path)
    trial = plant._default_trial(None)

    dt = float(plant.DT)
    settle_steps = int(round(float(plant.SETTLE_DURATION_S) / dt))
    duration_s = float(args.duration_sec) if args.duration_sec is not None else float(trial["duration_s"])
    if duration_s <= 0.0:
        raise ValueError("duration-sec must be positive")
    trial_steps = int(round(duration_s / dt)) + 1
    start_index = int(round(WINDOW_START_S / dt))
    sample_indices = set(range(start_index, trial_steps, STEP_STRIDE))
    slowmo = STEP_STRIDE * dt * float(args.fps)

    recorder = _RolloutRecorder(
        plant, repository_root, args.width, args.height, settle_steps, sample_indices
    )

    # Drive the SINGLE authoritative rollout; capture and render its live states.
    original_step = mujoco.mj_step
    mujoco.mj_step = recorder.hook(original_step)
    try:
        result = plant.run_trial(params, trial, record=True)
    finally:
        mujoco.mj_step = original_step

    traces = result["traces"]
    events = result["events"]
    summary = result["summary"]
    diagnostics = result["diagnostics"]
    phase_names = list(plant.PHASE_NAMES)
    time_s = [float(x) for x in traces["time_s"]]
    phase_index = [int(x) for x in traces["phase_index"]]
    fz_total = [float(x) for x in traces["fz_total_N"]]
    root_z = [float(x) for x in traces["root_z_m"]]
    bar_z = [float(x) for x in traces["bar_z_m"]]
    bar_vel = [float(x) for x in traces["bar_velocity_m_s"]]
    bar_disp = [float(x) for x in traces["bar_displacement_m"]]
    n_trace = len(time_s)

    # Per-region force-plate distribution + CoP for the heel->forefoot->toe rollover
    # dashboard. Read directly from the authoritative rollout telemetry channels
    # (no recomputation): region Fz shares, CoP_x, and the dominant loaded region.
    _fcg = list(plant.FOOT_CONTACT_GEOMS)
    _have_regions = f"{_fcg[0]}_fz_N" in traces
    cop_x_series = [None] * n_trace
    grf_dist: list[tuple[float, float, float, str]] = []
    if _have_regions:
        cop_raw = traces.get("cop_x_m")
        for i in range(n_trace):
            heel = sum(float(traces[f"{g}_fz_N"][i]) for g in _fcg if "heel" in g)
            fore = sum(float(traces[f"{g}_fz_N"][i]) for g in _fcg if "forefoot" in g)
            toe = sum(float(traces[f"{g}_fz_N"][i]) for g in _fcg if "toe" in g)
            tot = heel + fore + toe
            if tot > 1.0:
                dom = max((("HEEL", heel), ("FORE", fore), ("TOE", toe)), key=lambda kv: kv[1])[0]
                grf_dist.append((heel / tot, fore / tot, toe / tot, dom))
            else:
                grf_dist.append((0.0, 0.0, 0.0, "AIR"))
            if cop_raw is not None and cop_raw[i] is not None:
                cop_x_series[i] = float(cop_raw[i])
    else:
        grf_dist = [(0.0, 0.0, 0.0, "NA")] * n_trace
    grf_history = {
        "time_s": time_s,
        "heel_share": [float(x[0]) for x in grf_dist],
        "fore_share": [float(x[1]) for x in grf_dist],
        "toe_share": [float(x[2]) for x in grf_dist],
    }

    com_z_series = [float(p[2]) for p in recorder.com_points]
    bodyweight_N = float(summary.get(
        "quiet_baseline_mean_N",
        (float(params["body_mass_kg"]) + float(trial["external_load_kg"])) * float(plant.G),
    ))
    pos_arr = np.asarray(recorder.pos_samples, dtype=np.float64)
    vel_arr = np.asarray(recorder.vel_samples, dtype=np.float64)

    # Event indices/times (apex = highest COM between takeoff and landing).
    takeoff_index = int(events["takeoff_index"])
    landing_index = events.get("landing_index")
    landing_index = int(landing_index) if landing_index is not None else n_trace - 1
    apex_index = takeoff_index + int(np.argmax(com_z_series[takeoff_index:landing_index + 1])) \
        if landing_index > takeoff_index else takeoff_index
    event_times = {
        "takeoff_time_s": float(events["takeoff_time_s"]),
        "landing_time_s": float(events["landing_time_s"]) if events.get("landing_time_s") is not None else None,
        "apex_time_s": float(time_s[apex_index]),
        "movement_onset_time_s": float(events["movement_onset_time_s"]),
    }

    # Phase spans over the shown window (contiguous runs of one phase index).
    window = (time_s[start_index], time_s[min(trial_steps - 1, n_trace - 1)])
    phase_spans: list[tuple[float, float, str]] = []
    run_start = start_index
    for i in range(start_index + 1, n_trace):
        if phase_index[i] != phase_index[i - 1]:
            phase_spans.append((time_s[run_start], time_s[i], phase_names[phase_index[run_start]]))
            run_start = i
    phase_spans.append((time_s[run_start], window[1], phase_names[phase_index[run_start]]))

    # Lower-middle graph data. Primary curve is exact measured force-plate Fz.
    # Secondary curve is exact-state COM inertial force from central finite
    # differences of MuJoCo COM_z: F_COM,z = m_system * (a_COM,z + g).
    system_mass_kg = recorder.system_mass_kg or (bodyweight_N / float(plant.G))
    com_accel_z = _central_second_derivative(time_s, com_z_series)
    com_inertial_fz = [
        float(system_mass_kg * (a + float(plant.G))) for a in com_accel_z
    ]
    force_graph = {
        "time_s": time_s,
        "fz_total_N": fz_total,
        "bodyweight_N": bodyweight_N,
        "com_inertial_fz_N": com_inertial_fz,
    }
    fz_left_graph = [float(x) for x in traces.get("fz_left_N", [])]
    fz_right_graph = [float(x) for x in traces.get("fz_right_N", [])]
    if len(fz_left_graph) == len(time_s) and len(fz_right_graph) == len(time_s):
        force_graph["fz_left_N"] = fz_left_graph
        force_graph["fz_right_N"] = fz_right_graph

    joint_panel: dict[str, Any] | None = None
    joint_panel_items: list[dict[str, Any]] = []
    if pos_arr.ndim == 2 and vel_arr.ndim == 2:
        for label, joint_names in JOINT_PANEL_ITEMS:
            maps = [recorder.joint_index_map.get(name, {}) for name in joint_names]
            if not maps or not all(m.get("available") for m in maps):
                continue
            qpos_idx = [int(m["qpos_index"]) for m in maps]
            qvel_idx = [int(m["qvel_index"]) for m in maps]
            if max(qpos_idx) >= pos_arr.shape[1] or max(qvel_idx) >= vel_arr.shape[1]:
                continue
            if len(joint_names) == 2 and joint_names[0].startswith("left_") and joint_names[1].startswith("right_"):
                series = []
                for side_label, qpi, qvi in (("L", qpos_idx[0], qvel_idx[0]), ("R", qpos_idx[1], qvel_idx[1])):
                    series.append({
                        "side": side_label,
                        "q": [float(x) for x in pos_arr[:, qpi]],
                        "qdot": [float(x) for x in vel_arr[:, qvi]],
                    })
                joint_panel_items.append({
                    "label": label,
                    "joints": list(joint_names),
                    "series": series,
                    "qpos_indices": qpos_idx,
                    "qvel_indices": qvel_idx,
                })
            else:
                q = np.mean(pos_arr[:, qpos_idx], axis=1)
                qdot = np.mean(vel_arr[:, qvel_idx], axis=1)
                joint_panel_items.append({
                    "label": label,
                    "joints": list(joint_names),
                    "q": [float(x) for x in q],
                    "qdot": [float(x) for x in qdot],
                    "qpos_indices": qpos_idx,
                    "qvel_indices": qvel_idx,
                })
    if joint_panel_items:
        joint_panel = {"time_s": time_s, "items": joint_panel_items}

    # Which sampled frames should carry an event flash.
    flash_frames = {takeoff_index: f"TAKEOFF  V = {_fmt(float(summary['takeoff_velocity_m_s']), 2)} M/S",
                    apex_index: f"APEX  H = {_fmt(float(summary['jump_height_im_m']))} M",
                    landing_index: "LANDING"}

    def _nearest_flash(idx: int) -> str | None:
        for target, label in flash_frames.items():
            if abs(idx - target) <= STEP_STRIDE // 2 + 1:
                return label
        return None

    output_file = args.output_file.resolve() if args.output_file else output_dir / "loaded_cmj_demo.mp4"
    proc = _open_ffmpeg(ffmpeg, output_file, args.width, args.height, args.fps)
    frame_records: list[dict[str, Any]] = []
    ordered = sorted(recorder.frames, key=lambda fr: fr["trial_index"])
    try:
        for fr in ordered:
            idx = fr["trial_index"]
            frame = fr["rgb"]
            _compose_overlay(
                frame, t_s=time_s[idx], phase=phase_names[phase_index[idx]], fz_N=fz_total[idx],
                root_z=root_z[idx], com_z=com_z_series[idx], bar_z=bar_z[idx],
                bar_vel=bar_vel[idx], bar_disp=bar_disp[idx],
                bodyweight_N=bodyweight_N, window=window, phase_spans=phase_spans,
                events=event_times, slowmo=slowmo, event_flash=_nearest_flash(idx),
                grf=grf_dist[idx], grf_history=grf_history, cop_x=cop_x_series[idx],
                force_graph=force_graph, joint_panel=joint_panel,
                current_index=idx, com_screen_path=fr.get("com_screen_path"),
                force_plate_screen_xy=fr.get("force_plate_screen_xy"),
                landmark_screen=fr.get("landmark_screen"),
                lpt_screen_xy=fr.get("lpt_screen_xy"),
                barbell_screen_xy=fr.get("barbell_screen_xy"),
            )
            proc.stdin.write(frame.tobytes())
            com_path = fr.get("com_screen_path") or []
            com_screen_xy = list(com_path[-1]) if com_path else None
            right_plot_x0, right_plot_x1 = _right_time_plot_bounds(args.width)
            frame_records.append({
                "trial_index": idx, "time_s": time_s[idx], "phase": phase_names[phase_index[idx]],
                "phase_index": phase_index[idx], "fz_total_N": fz_total[idx], "root_z_m": root_z[idx],
                "com_z_m": com_z_series[idx], "bar_z_m": bar_z[idx],
                "any_contact": bool(traces["left_foot_contact"][idx] or traces["right_foot_contact"][idx]),
                "com_screen_xy": com_screen_xy,
                "force_plate_screen_xy": list(fr["force_plate_screen_xy"])
                if fr.get("force_plate_screen_xy") is not None else None,
                "lpt_screen_xy": list(fr["lpt_screen_xy"])
                if fr.get("lpt_screen_xy") is not None else None,
                "barbell_screen_xy": list(fr["barbell_screen_xy"])
                if fr.get("barbell_screen_xy") is not None else None,
                "landmark_screen_count": len(fr.get("landmark_screen") or []),
                "right_graph_plot_bounds": {
                    "grf_trace_plot_x0": right_plot_x0,
                    "grf_trace_plot_x1": right_plot_x1,
                    "fz_graph_plot_x0": right_plot_x0,
                    "fz_graph_plot_x1": right_plot_x1,
                },
            })
        last = ordered[-1]["rgb"] if ordered else None
        for _ in range(END_HOLD_FRAMES):
            if last is not None:
                proc.stdin.write(last.tobytes())
    finally:
        proc.stdin.close()
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with code {ret}")

    rendered_frame_count = len(ordered) + (END_HOLD_FRAMES if ordered else 0)

    # Provenance: hash the exact recorded state series of the authoritative rollout.
    pos_sha = hashlib.sha256(np.ascontiguousarray(pos_arr).tobytes()).hexdigest()
    vel_sha = hashlib.sha256(np.ascontiguousarray(vel_arr).tobytes()).hexdigest()
    # Cross-check: the states we captured and rendered reproduce the authority trace.
    root_z_diff = float(np.max(np.abs(np.asarray(recorder.root_z_world[:n_trace]) - np.asarray(root_z[:len(recorder.root_z_world)])))) if recorder.root_z_world else 0.0

    per_frame_state = []
    for fr in ordered:
        idx = fr["trial_index"]
        if idx < len(recorder.pos_samples):
            per_frame_state.append({
                "trial_index": idx,
                "pos_sha256": hashlib.sha256(np.ascontiguousarray(recorder.pos_samples[idx]).tobytes()).hexdigest(),
                "vel_sha256": hashlib.sha256(np.ascontiguousarray(recorder.vel_samples[idx]).tobytes()).hexdigest(),
            })

    makehuman_meta = recorder.makehuman.metadata() if recorder.makehuman is not None else {
        "available": False,
        "load_error": "MakeHuman renderer was not initialized",
        "variant": "unavailable",
        "render_only": True,
    }
    provenance = {
        "qpos_trace_available": True,
        "qvel_trace_available": True,
        "rendered_frame_count": rendered_frame_count,
        "state_sample_count": int(pos_arr.shape[0]),
        "ndof": int(pos_arr.shape[1]) if pos_arr.ndim == 2 else 0,
        "frame_sample_indices": [int(fr["trial_index"]) for fr in ordered],
        "qpos_trace_sha256": pos_sha,
        "qvel_trace_sha256": vel_sha,
        "source_model": "assets/loaded_cmj_model.xml via plant.build_model -> mujoco.MjModel.from_xml_path",
        "motion_source": "plant.run_trial mujoco.mj_step rollout (read-only instrumentation)",
        "render_model_xml_used": bool(makehuman_meta.get("available", False)),
        "render_model_xml_source": str(makehuman_meta.get("asset_xml",
            (repository_root / MAKEHUMAN_SKIN_RELATIVE_XML).resolve())),
        "human_render_is_true_makehuman_skin": bool(recorder.human_is_skin),
        "makehuman_visual_overlay": makehuman_meta,
        "makehuman_visual_used_in_render": bool(recorder.human_visual_used),
        "makehuman_mesh_composited": bool(makehuman_meta.get("available", False)),
        "makehuman_obj_used": bool(makehuman_meta.get("makehuman_obj_used", makehuman_meta.get("obj_used", False))),
        "makehuman_obj_used_as_primary_human": bool(
            makehuman_meta.get("available", False) and not recorder.human_is_skin
            and bool(makehuman_meta.get("obj_used", False))),
        "mujoco_native_skin_used": bool(makehuman_meta.get("mujoco_native_skin_used", False)),
        "rigged_makehuman_scene_skin_used": bool(makehuman_meta.get("rigged_makehuman_scene_skin_used", False)),
        "segmented_obj_primary_human": bool(makehuman_meta.get("segmented_obj_primary_human",
            makehuman_meta.get("available", False) and not recorder.human_is_skin)),
        "procedural_capsule_human": bool(makehuman_meta.get("procedural_capsule_human",
            not makehuman_meta.get("available", False))),
        "makehuman_render_only": True,
        "skin_fallback_reason": recorder.skin_fallback_reason,
        "makehuman_dae_vendored": bool((repository_root / MAKEHUMAN_DAE_RELATIVE_PATH).exists()),
        "plant_loads_makehuman_asset": False,
        "data_generator_loads_makehuman_asset": False,
        "separate_render_dynamics_used": False,
        "separate_render_mj_forward_only": False,
        "makehuman_visual_qpos_set_render_only": False,
        "makehuman_visual_qpos_is_dynamics_plant": False,
        "plant_qpos_qvel_replay_after_init": bool(diagnostics.get("qpos_qvel_write_after_init", False)),
        "qpos_qvel_write_after_init": bool(diagnostics.get("qpos_qvel_write_after_init", False)),
        "qfrc_applied_norm_max": float(diagnostics.get("qfrc_applied_norm_max", 0.0)),
        "xfrc_applied_norm_max": float(diagnostics.get("xfrc_applied_norm_max", 0.0)),
        "secondary_flight_count": int(diagnostics.get("secondary_flight_count", 0)),
        "provenance_cross_check_root_z_max_abs_diff_m": root_z_diff,
        "takeoff_velocity_m_s": float(summary["takeoff_velocity_m_s"]),
        "jump_height_im_m": float(summary["jump_height_im_m"]),
        "root_z_airborne_peak_gain_m": float(summary["root_z_airborne_peak_gain_m"]),
        "takeoff_index": takeoff_index, "apex_index": int(apex_index), "landing_index": landing_index,
        "com_foreground_method": "post-render screen-space overlay from exact MuJoCo COM projection",
        "force_graph_primary_curve": "Force-plate Fz-time curve",
        "force_graph_secondary_curve": "COM inertial Fz = m_system * (a_COM,z + g)",
        "system_mass_kg_for_com_inertial": float(system_mass_kg),
        "right_graph_plot_bounds_1280": {
            "grf_trace_plot_x0": _right_time_plot_bounds(args.width)[0],
            "grf_trace_plot_x1": _right_time_plot_bounds(args.width)[1],
            "fz_graph_plot_x0": _right_time_plot_bounds(args.width)[0],
            "fz_graph_plot_x1": _right_time_plot_bounds(args.width)[1],
            "grf_panel_x0": _right_dashboard_bounds(args.width)[0],
            "grf_panel_x1": _right_dashboard_bounds(args.width)[1],
            "fz_bw_meter_x0": _right_dashboard_bounds(args.width)[2],
            "fz_bw_meter_x1": _right_dashboard_bounds(args.width)[3],
            "fz_graph_outer_x0": _right_time_plot_bounds(args.width)[0],
            "fz_graph_outer_x1": _right_time_plot_bounds(args.width)[1],
        },
        "left_joint_panel_bounds_1280": {
            "x0": LEFT_JOINT_PANEL_X0,
            "x1": LEFT_JOINT_PANEL_X1,
            "y0": LEFT_JOINT_PANEL_Y0,
            "y1": args.height - LEFT_JOINT_PANEL_BOTTOM_MARGIN_PX,
        },
        "lower_left_telemetry_bounds_1280": {
            "x0": LOWER_LEFT_TELEMETRY_BOUNDS[0],
            "x1": LOWER_LEFT_TELEMETRY_BOUNDS[2],
            "y0": args.height - LOWER_LEFT_TELEMETRY_BOUNDS[1],
            "y1": args.height - LOWER_LEFT_TELEMETRY_BOUNDS[3],
        },
        "lpt_scene_label_text": LPT_SCENE_LABEL,
        "lpt_tether_telemetry_text": LPT_TETHER_TELEMETRY_LABEL,
        "joint_index_map": recorder.joint_index_map,
        "joint_panel_displayed": [
            {"label": item["label"], "joints": item["joints"],
             "qpos_indices": item["qpos_indices"], "qvel_indices": item["qvel_indices"]}
            for item in joint_panel_items
        ],
        "render_frame_records": frame_records,
        "per_frame_state_hashes": per_frame_state,
    }
    (output_dir / "render_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")

    if args.telemetry_json is not None:
        telemetry = {
            "video": {"width": args.width, "height": args.height, "fps": args.fps,
                      "frame_count": rendered_frame_count, "slowmo_factor": slowmo},
            "window_s": list(window), "phase_spans": [[a, b, n] for a, b, n in phase_spans],
            "events": event_times, "summary": {k: float(summary[k]) for k in summary},
            "provenance": {k: provenance[k] for k in (
                "qpos_trace_available", "qvel_trace_available", "qpos_trace_sha256",
                "qvel_trace_sha256", "render_model_xml_used", "render_model_xml_source",
                "makehuman_visual_overlay", "makehuman_visual_used_in_render",
                "makehuman_mesh_composited", "makehuman_obj_used",
                "rigged_makehuman_scene_skin_used", "makehuman_dae_vendored",
                "separate_render_dynamics_used",
                "qfrc_applied_norm_max", "xfrc_applied_norm_max", "secondary_flight_count",
                "com_foreground_method", "force_graph_primary_curve", "force_graph_secondary_curve",
                "system_mass_kg_for_com_inertial", "right_graph_plot_bounds_1280",
                "left_joint_panel_bounds_1280", "lower_left_telemetry_bounds_1280",
                "lpt_scene_label_text", "lpt_tether_telemetry_text",
                "joint_panel_displayed", "render_frame_records")},
            "frames": frame_records,
        }
        args.telemetry_json.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.telemetry_json.resolve().write_text(
            json.dumps(telemetry, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")

    print(f"Wrote reviewer rendering to {output_file} ({rendered_frame_count} frames, "
          f"slow-motion {slowmo:.2f}x)")
    print(f"Provenance: qpos_sha={pos_sha[:16]} qvel_sha={vel_sha[:16]} "
          f"root_z_cross_check_max_abs_diff={root_z_diff:.3e} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
