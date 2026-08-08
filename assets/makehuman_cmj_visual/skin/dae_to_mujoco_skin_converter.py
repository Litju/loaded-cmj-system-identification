#!/usr/bin/env python3
"""Convert the repository-local MakeHuman COLLADA (.dae) body into a MuJoCo-native,
render-only linear-blend-skinning (LBS) ``<skin>`` asset.

This is a *provenance / reproducibility* tool. It reads only the repo-local
MakeHuman DAE (CC0 base mesh + default rig) and emits:

  * ``makehuman_skin_visual.xml``   -- a standalone, render-only MuJoCo model:
        163 mocap "bone" bodies + one ``<asset><skin>`` bound to them. The skin
        carries the full continuous body surface (13,380 verts, 26,756 tris).
        No collision geoms, no contacts, no actuators -> physics-inert.
  * ``makehuman_skin_bones.json`` -- per-bone bind world transform (pos+quat),
        bind world position, parent sid, plus mesh/rig provenance counts. The
        renderer uses this to drive the bones from live CMJ plant landmarks.
  * ``makehuman_skin_manifest.json`` -- summary counts + integrity hashes.

MuJoCo LBS convention (verified against installed MuJoCo 3.8.0):
    v_world = sum_b w_b * BodyXform_b * (BindXform_b^{-1} * v_skin)
COLLADA LBS convention:
    v_world = sum_b w_b * JointWorld_b * InvBind_b * (BindShape * v_pos)
Matching the two at the bind pose (BodyXform_b == BindXform_b) gives:
    v_skin      = BindShape * v_pos
    BindXform_b = InvBind_b^{-1}   (bone world transform in the bind pose)
so setting a bone body's world transform away from BindXform_b deforms the skin.
"""
from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
HERE = Path(__file__).resolve().parent
ASSET_ROOT = HERE.parent  # .../makehuman_cmj_visual
DAE_PATH = ASSET_ROOT / "makehuman_cmj_body.dae"

# Skin-tone soft-tissue colour (shared with the renderer's MAKEHUMAN_SKIN_RGBA).
SKIN_RGBA = (0.79, 0.58, 0.45, 1.0)


def _text_floats(elem: ET.Element) -> np.ndarray:
    return np.array(elem.text.split(), dtype=np.float64)


def _mat_to_pos_quat(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4x4 homogeneous (row-major) -> (pos[3], quat[wxyz]).

    Assumes a rigid (rotation only, no scale/shear) upper-left 3x3, which holds
    for MakeHuman inverse-bind matrices. Uses a numerically stable conversion.
    """
    pos = m[:3, 3].copy()
    r = m[:3, :3].copy()
    # Re-orthonormalise defensively against tiny float drift.
    u, _, vt = np.linalg.svd(r)
    r = u @ vt
    if np.linalg.det(r) < 0:  # guard against a reflected basis
        u[:, -1] *= -1.0
        r = u @ vt
    t = np.trace(r)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return pos, q


def parse_dae(dae_path: Path) -> dict:
    root = ET.parse(dae_path).getroot()

    def find(base, path):
        return base.find(path, NS)

    def findall(base, path):
        return base.findall(path, NS)

    # --- base skin controller -------------------------------------------------
    skin = None
    for ctrl in findall(root, ".//c:controller"):
        s = find(ctrl, "c:skin")
        if s is not None and "base-skin" in (ctrl.get("id") or ""):
            skin = s
            break
    if skin is None:
        raise RuntimeError("base-skin controller not found in DAE")

    bind_shape = _text_floats(find(skin, "c:bind_shape_matrix")).reshape(4, 4)

    # sources keyed by id
    sources: dict[str, ET.Element] = {s.get("id"): s for s in findall(skin, "c:source")}

    def source_of(inputs, semantic):
        for inp in inputs:
            if inp.get("semantic") == semantic:
                return inp.get("source").lstrip("#")
        raise KeyError(semantic)

    joints_el = find(skin, "c:joints")
    j_inputs = findall(joints_el, "c:input")
    joint_src = sources[source_of(j_inputs, "JOINT")]
    invbind_src = sources[source_of(j_inputs, "INV_BIND_MATRIX")]

    _jarr = find(joint_src, "c:IDREF_array")
    if _jarr is None:
        _jarr = find(joint_src, "c:Name_array")
    joint_sids = _jarr.text.split()
    n_joints = len(joint_sids)
    invbind = _text_floats(find(invbind_src, "c:float_array")).reshape(n_joints, 4, 4)

    # --- per-vertex weights ---------------------------------------------------
    vw = find(skin, "c:vertex_weights")
    vw_inputs = findall(vw, "c:input")
    weight_src = sources[source_of(vw_inputs, "WEIGHT")]
    weights = _text_floats(find(weight_src, "c:float_array"))
    # offsets
    off = {inp.get("semantic"): int(inp.get("offset")) for inp in vw_inputs}
    stride = max(off.values()) + 1
    vcount = np.array(vw.find("c:vcount", NS).text.split(), dtype=int)
    vindex = np.array(vw.find("c:v", NS).text.split(), dtype=int)
    n_verts = int(vw.get("count"))
    assert len(vcount) == n_verts

    # --- base mesh geometry ---------------------------------------------------
    geo = None
    for g in findall(root, ".//c:geometry"):
        if "baseMesh" in (g.get("id") or ""):
            geo = g
            break
    mesh = find(geo, "c:mesh")
    # positions live in the source referenced by <vertices><input semantic=POSITION>
    verts_el = find(mesh, "c:vertices")
    pos_src_id = source_of(findall(verts_el, "c:input"), "POSITION")
    pos_src = None
    for s in findall(mesh, "c:source"):
        if s.get("id") == pos_src_id:
            pos_src = s
            break
    positions = _text_floats(find(pos_src, "c:float_array")).reshape(-1, 3)
    assert positions.shape[0] == n_verts, (positions.shape, n_verts)

    polylist = find(mesh, "c:polylist")
    pl_inputs = findall(polylist, "c:input")
    pl_stride = max(int(i.get("offset")) for i in pl_inputs) + 1
    vertex_offset = int(source_of(pl_inputs, "VERTEX") and
                        [i for i in pl_inputs if i.get("semantic") == "VERTEX"][0].get("offset"))
    poly_vcount = np.array(polylist.find("c:vcount", NS).text.split(), dtype=int)
    p = np.array(polylist.find("c:p", NS).text.split(), dtype=int).reshape(-1, pl_stride)
    vert_index_stream = p[:, vertex_offset]

    return dict(
        bind_shape=bind_shape,
        joint_sids=joint_sids,
        invbind=invbind,
        weights=weights,
        weight_offset=off["WEIGHT"],
        joint_offset=off["JOINT"],
        stride=stride,
        vcount=vcount,
        vindex=vindex,
        positions=positions,
        poly_vcount=poly_vcount,
        vert_index_stream=vert_index_stream,
        n_verts=n_verts,
        n_joints=n_joints,
    )


def build_joint_hierarchy(dae_path: Path, joint_sids: list[str]) -> dict[str, str | None]:
    """Return {sid: parent_sid} from the visual_scene JOINT node tree."""
    root = ET.parse(dae_path).getroot()
    parent: dict[str, str | None] = {}

    def walk(node, parent_sid):
        sid = node.get("sid")
        is_joint = node.get("type") == "JOINT"
        here = sid if is_joint else parent_sid
        if is_joint and sid is not None:
            parent[sid] = parent_sid
        for child in node.findall("c:node", NS):
            walk(child, here)

    for scene in root.findall(".//c:visual_scene", NS):
        for node in scene.findall("c:node", NS):
            walk(node, None)
    return {s: parent.get(s) for s in joint_sids}


def convert() -> dict:
    data = parse_dae(DAE_PATH)
    bind_shape = data["bind_shape"]
    positions = data["positions"]
    n_verts = data["n_verts"]
    n_joints = data["n_joints"]

    # Skin rest vertices in world = BindShape * position.
    homog = np.concatenate([positions, np.ones((n_verts, 1))], axis=1)
    skin_verts = (bind_shape @ homog.T).T[:, :3]

    # Bone bind world transforms = inverse of inverse-bind matrix.
    bind_pos = np.zeros((n_joints, 3))
    bind_quat = np.zeros((n_joints, 4))
    for j in range(n_joints):
        world = np.linalg.inv(data["invbind"][j])
        p, q = _mat_to_pos_quat(world)
        bind_pos[j] = p
        bind_quat[j] = q

    # Per-bone (vertid, weight) lists from the interleaved v stream.
    weights = data["weights"]
    vindex = data["vindex"]
    stride = data["stride"]
    jo = data["joint_offset"]
    wo = data["weight_offset"]
    bone_vertid: list[list[int]] = [[] for _ in range(n_joints)]
    bone_vertweight: list[list[float]] = [[] for _ in range(n_joints)]
    cursor = 0
    for v in range(n_verts):
        k = data["vcount"][v]
        block = vindex[cursor: cursor + k * stride].reshape(k, stride)
        cursor += k * stride
        for row in block:
            j = int(row[jo])
            w = float(weights[int(row[wo])])
            if w <= 0.0:
                continue
            bone_vertid[j].append(v)
            bone_vertweight[j].append(w)
    assert cursor == len(vindex)

    # Triangulate the polygon stream (all quads here, but handle general n-gons).
    tris: list[tuple[int, int, int]] = []
    stream = data["vert_index_stream"]
    off = 0
    for k in data["poly_vcount"]:
        face = stream[off: off + k]
        off += k
        for t in range(1, k - 1):
            tris.append((int(face[0]), int(face[t]), int(face[t + 1])))
    faces = np.array(tris, dtype=int)

    parents = build_joint_hierarchy(DAE_PATH, data["joint_sids"])

    return dict(
        skin_verts=skin_verts,
        faces=faces,
        bind_pos=bind_pos,
        bind_quat=bind_quat,
        bone_vertid=bone_vertid,
        bone_vertweight=bone_vertweight,
        joint_sids=data["joint_sids"],
        parents=parents,
        n_verts=n_verts,
        n_joints=n_joints,
    )


def _fmt_floats(a: np.ndarray, prec: int = 5) -> str:
    return " ".join(f"{x:.{prec}f}" for x in a.ravel())


def _fmt_ints(a) -> str:
    return " ".join(str(int(x)) for x in a)


def write_assets(conv: dict) -> dict:
    n_joints = conv["n_joints"]
    sids = conv["joint_sids"]

    def body_name(sid: str) -> str:
        return f"mhb_{sid}"

    # --- skin XML -------------------------------------------------------------
    lines = []
    lines.append('<mujoco model="makehuman_cmj_skin_visual">')
    lines.append('  <compiler angle="radian"/>')
    lines.append('  <visual>')
    lines.append('    <global offwidth="1280" offheight="720"/>')
    lines.append('    <quality shadowsize="2048"/>')
    lines.append('    <headlight ambient="0.55 0.55 0.55" diffuse="0.62 0.62 0.62" '
                 'specular="0.08 0.08 0.08"/>')
    lines.append('  </visual>')
    lines.append('  <asset>')
    r, g, b, a = SKIN_RGBA
    verts_str = _fmt_floats(conv["skin_verts"], 5)
    faces_str = _fmt_ints(conv["faces"].ravel())
    lines.append(f'    <skin name="mh_skin" rgba="{r} {g} {b} {a}" inflate="0.0"')
    lines.append(f'          vertex="{verts_str}"')
    lines.append(f'          face="{faces_str}">')
    for j in range(n_joints):
        if not conv["bone_vertid"][j]:
            # A bone with no influenced vertices still must not be emitted as a
            # skin <bone> (MuJoCo requires >=1 vertid). Skip it here; its body is
            # still created so the hierarchy/retarget stays complete.
            continue
        bp = _fmt_floats(conv["bind_pos"][j], 6)
        bq = _fmt_floats(conv["bind_quat"][j], 6)
        vid = _fmt_ints(conv["bone_vertid"][j])
        vwt = _fmt_floats(np.array(conv["bone_vertweight"][j]), 5)
        lines.append(f'    <bone body="{body_name(sids[j])}" bindpos="{bp}" '
                     f'bindquat="{bq}"')
        lines.append(f'          vertid="{vid}"')
        lines.append(f'          vertweight="{vwt}"/>')
    lines.append('  </skin>')
    lines.append('  </asset>')
    lines.append('  <worldbody>')
    # 163 render-only "bone" bodies at their bind world transform. Each carries a
    # free joint so the render driver sets the bone world pose by writing that
    # joint's qpos (pos+quat) and calling mj_forward for rendering only -- the
    # same render-only visual-posing pattern used by the segmented fallback model.
    # A vanishing inertial keeps the free body well-formed; this model is never
    # stepped (mj_forward only), so no dynamics act on it.
    for j in range(n_joints):
        bp = _fmt_floats(conv["bind_pos"][j], 6)
        bq = _fmt_floats(conv["bind_quat"][j], 6)
        nm = body_name(sids[j])
        lines.append(f'    <body name="{nm}" pos="{bp}" quat="{bq}">')
        lines.append(f'      <freejoint name="{nm}_free"/>')
        lines.append('      <inertial pos="0 0 0" mass="0.0001" '
                     'diaginertia="1e-06 1e-06 1e-06"/>')
        lines.append('    </body>')
    lines.append('  </worldbody>')
    lines.append('</mujoco>')
    xml_text = "\n".join(lines) + "\n"
    xml_path = HERE / "makehuman_skin_visual.xml"
    xml_path.write_text(xml_text, encoding="utf-8")

    # --- bones JSON (retarget driver reads this) ------------------------------
    bones = []
    for j in range(n_joints):
        bones.append(dict(
            sid=sids[j],
            body=body_name(sids[j]),
            parent=conv["parents"].get(sids[j]),
            bind_pos=conv["bind_pos"][j].tolist(),
            bind_quat=conv["bind_quat"][j].tolist(),
            n_vert=len(conv["bone_vertid"][j]),
        ))
    bones_path = HERE / "makehuman_skin_bones.json"
    bones_path.write_text(json.dumps(dict(
        source_dae="makehuman_cmj_body.dae",
        n_joints=n_joints,
        skin_body_prefix="mhb_",
        bones=bones,
    ), indent=2), encoding="utf-8")

    # --- manifest -------------------------------------------------------------
    manifest = dict(
        generator="dae_to_mujoco_skin_converter.py",
        source_dae="makehuman_cmj_body.dae",
        skin_xml="makehuman_skin_visual.xml",
        bones_json="makehuman_skin_bones.json",
        n_vertices=int(conv["n_verts"]),
        n_faces=int(conv["faces"].shape[0]),
        n_bones_total=n_joints,
        n_bones_skinned=int(sum(1 for j in range(n_joints) if conv["bone_vertid"][j])),
        skin_rgba=list(SKIN_RGBA),
        lbs_convention="v_world = sum_b w_b * BodyXform_b * inv(BindXform_b) * v_skin",
        render_only=True,
        physics_inert=True,
        xml_sha256=hashlib.sha256(xml_text.encode()).hexdigest(),
    )
    (HERE / "makehuman_skin_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    conv = convert()
    manifest = write_assets(conv)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
