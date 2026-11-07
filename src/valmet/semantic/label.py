"""Name every segmented object with Gemini vision (zero-shot, fixed vocabulary).

Each object gets a 3-panel render from the cleaned Gaussians (object front | object side |
context with red box). Batches go to Gemini, which returns {label, confidence, reason}.
Low-confidence or failed answers fall back to geometric rules. Answers are cached per object
signature, so re-runs only ask about changed objects.

Ported from the field_copilot demo (PR #5) to the shared world frame (+Z up, meters).

Inputs : <scene_dir>/objects.json, semantic_details.json, semantic/segment_points.npz,
         scene_clean.npz
Outputs: objects.json (label, confidence), semantic_details.json (label_source, label_reason,
         label_vlm), semantic/crops/obj_xxxx.png, semantic/label_cache.json,
         reports/preview/<scene_id>_label_contact_sheet.png

Usage:
    python -m valmet.semantic.label --scene-dir data/processed/equipment_room [--rules-only]
"""

from __future__ import annotations

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from valmet.common.artifacts import OBJECTS_FILE, REPO_ROOT, load_list, save_list
from valmet.common.contracts import DetectedObject
from valmet.preprocessing.export import SCENE_CLEAN_FILE
from valmet.semantic.segment import DETAILS_FILE, SEGMENT_POINTS_FILE, SEMANTIC_DIR

CROPS_DIR = f"{SEMANTIC_DIR}/crops"
CACHE_FILE = f"{SEMANTIC_DIR}/label_cache.json"
RAW_DIR = f"{SEMANTIC_DIR}/label_raw"

VOCAB = (
    "pump",
    "motor",
    "control valve",
    "manual valve/handwheel",
    "pressure transmitter/gauge",
    "vibration sensor",
    "pipe segment",
    "pipe fitting/flange",
    "tank/vessel",
    "electrical cabinet",
    "cable tray",
    "ladder/stairs",
    "cage/rack",
    "handrail/railing",
    "door",
    "sign",
    "box/crate",
    "trolley",
    "light fixture",
    "duct",
    "wall",
    "floor",
    "other",
)


@dataclass(frozen=True)
class LabelParams:
    batch_size: int = 30  # objects per Gemini call (free tier: ~20 requests/day/model)
    workers: int = 2
    min_conf: float = 0.45  # below this -> geometry-rule label (Gemini guess kept as label_vlm)
    rules_conf: float = 0.4
    panel_px: int = 320  # saved crop panel size
    send_px: int = 224  # panel size sent to Gemini
    splat_px: int = 2  # point radius in px
    context_margin: float = 0.69
    context_depth: float = 0.58  # context panel shows only points within this depth of the object
    background: tuple[int, int, int] = (235, 235, 235)
    thumb_px: int = 128
    sheet_cols: int = 16


# ---------- data ----------


def load_inputs(scene_dir: Path) -> tuple[list[DetectedObject], dict, np.ndarray, np.ndarray, np.ndarray]:
    """Objects, details doc, and the cleaned Gaussians with an object index per Gaussian."""
    objects = load_list(scene_dir / OBJECTS_FILE, DetectedObject)
    details = json.loads((scene_dir / DETAILS_FILE).read_text(encoding="utf-8"))
    seg = np.load(scene_dir / SEGMENT_POINTS_FILE)
    scene = np.load(scene_dir / SCENE_CLEAN_FILE)
    return (
        objects,
        details,
        scene["positions"].astype(np.float64),
        scene["rgbs"],
        seg["gaussian_labels"],
    )


# ---------- rendering ----------


def raster(
    pts: np.ndarray,
    rgb: np.ndarray,
    h_axis: int,
    depth_axis: int,
    lo: np.ndarray,
    hi: np.ndarray,
    px: int,
    splat_px: int,
    bg: tuple[int, int, int],
) -> Image.Image:
    """Orthographic z-buffered point render of the window [lo, hi] = (h, z), +Z up on screen.
    Larger depth-axis values are nearer."""
    img = np.full((px, px, 3), bg, np.uint8)
    if len(pts) == 0:
        return Image.fromarray(img)
    u, v = _to_px(pts[:, h_axis], pts[:, 2], lo, hi, px)
    depth = -pts[:, depth_axis]  # smaller = nearer
    offs = [(dx, dy) for dx in range(-splat_px, splat_px + 1) for dy in range(-splat_px, splat_px + 1)]
    U = np.concatenate([u + dx for dx, _ in offs]).astype(int)
    V = np.concatenate([v + dy for _, dy in offs]).astype(int)
    D = np.tile(depth, len(offs))
    C = np.tile((np.clip(rgb, 0, 1) * 255).astype(np.uint8), (len(offs), 1))
    ok = (U >= 0) & (U < px) & (V >= 0) & (V < px)
    U, V, D, C = U[ok], V[ok], D[ok], C[ok]
    order = np.argsort(D, kind="stable")
    flat = V[order] * px + U[order]
    _, first = np.unique(flat, return_index=True)
    sel = order[first]
    img[V[sel], U[sel]] = C[sel]
    return Image.fromarray(img)


def _to_px(h: Any, z: Any, lo: np.ndarray, hi: np.ndarray, px: int) -> tuple[Any, Any]:
    span = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.08 + 1e-6
    c = (lo + hi) / 2
    u = ((h - c[0]) / span + 0.5) * (px - 1)
    v = (px - 1) - ((z - c[1]) / span + 0.5) * (px - 1)
    return u, v


def object_panels(
    lo: np.ndarray,
    hi: np.ndarray,
    member: np.ndarray,
    xyz: np.ndarray,
    rgb: np.ndarray,
    lp: LabelParams,
) -> Image.Image:
    """[front (look +y) | side (look +x) | context front with red box]."""
    px, sp, bg = lp.panel_px, lp.splat_px, lp.background
    P, Cc = xyz[member], rgb[member]
    pad = np.maximum((hi - lo) * 0.05, 0.02)
    front = raster(P, Cc, 0, 1, (lo - pad)[[0, 2]], (hi + pad)[[0, 2]], px, sp, bg)
    side = raster(P, Cc, 1, 0, (lo - pad)[[1, 2]], (hi + pad)[[1, 2]], px, sp, bg)
    clo, chi = lo - lp.context_margin, hi + lp.context_margin
    dy = lp.context_depth  # only the object's depth slab, not the whole room behind it
    inwin = (
        (xyz[:, 0] >= clo[0])
        & (xyz[:, 0] <= chi[0])
        & (xyz[:, 2] >= clo[2])
        & (xyz[:, 2] <= chi[2])
        & (xyz[:, 1] >= lo[1] - dy)
        & (xyz[:, 1] <= hi[1] + dy)
    )
    wlo, whi = clo[[0, 2]], chi[[0, 2]]
    ctx = raster(xyz[inwin], rgb[inwin], 0, 1, wlo, whi, px, max(sp - 1, 1), bg)
    u0, v0 = _to_px(lo[0], hi[2], wlo, whi, px)
    u1, v1 = _to_px(hi[0], lo[2], wlo, whi, px)
    ImageDraw.Draw(ctx).rectangle([u0, v0, u1, v1], outline=(230, 0, 0), width=3)
    out = Image.new("RGB", (px * 3 + 8, px), (255, 255, 255))
    for k, im in enumerate((front, side, ctx)):
        out.paste(im, (k * (px + 4), 0))
    return out


# ---------- Gemini ----------

PROMPT = """You label objects extracted from a 3D Gaussian-splat scan of an industrial boiler room
(pipes, valves, pumps, ladders/cages, cabinets, doors). Each image has 3 panels: LEFT = the object alone,
front view; MIDDLE = the object alone, side view; RIGHT = the object in its surroundings (red box).
Renders come from a sparse noisy scan, so they look blurry. Real-world size and height are given.

Choose exactly one label per object from this list:
{vocab}
Use "other" if none fits; be honest with confidence (0-1). Return ONLY a JSON array:
[{{"id": "obj_0001", "label": "...", "confidence": 0.0, "reason": "max 12 words"}}]"""


def describe(object_id: str, d: dict) -> str:
    lo, hi = np.asarray(d["bbox_min"]), np.asarray(d["bbox_max"])
    e = hi - lo
    f = d["features"]
    return (
        f"{object_id}: size W{e[0]:.2f} x H{e[2]:.2f} x D{e[1]:.2f} m, bottom "
        f"{f['height_bottom']:.2f} m above floor, linearity {f['linearity']:.2f}, planarity "
        f"{f['planarity']:.2f}, geometry hint: {d['hint'] or 'none'}"
        f"{', overhead' if 'overhead' in d.get('tags', []) else ''}"
    )


def png_bytes(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def parse_list(text: str) -> list[dict]:
    t = text.strip()
    t = t[t.index("[") : t.rindex("]") + 1] if "[" in t else t
    res = json.loads(t)
    if isinstance(res, dict):
        res = next((v for v in res.values() if isinstance(v, list)), [res])
    return [r for r in res if isinstance(r, dict)]


def gemini_batch(
    batch: list[tuple[str, dict, Image.Image]], raw_dir: Path, lp: LabelParams
) -> tuple[list[dict], str]:
    from google.genai import types

    from valmet.semantic.gemini import generate

    parts: list[Any] = [PROMPT.format(vocab=", ".join(VOCAB))]
    for oid, d, im in batch:
        small = im.resize((lp.send_px * 3, lp.send_px))
        parts += [
            describe(object_id=oid, d=d),
            types.Part.from_bytes(data=png_bytes(small), mime_type="image/png"),
        ]
    text, model = generate(
        contents=parts,
        json_mode=True,
        temperature=0.1,
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{batch[0][0]}.txt").write_text(f"{model}\n{text}", encoding="utf-8")
    return parse_list(text), model


# ---------- rules fallback ----------


def label_by_rules(d: dict) -> str:
    f = d["features"]
    ext = float(np.max(np.asarray(d["bbox_max"]) - np.asarray(d["bbox_min"])))
    if d["hint"] == "pipe segment" or (f["linearity"] > 0.85 and ext > 0.46):
        return "pipe segment"
    if d["hint"] == "fitting/valve":
        return "pipe fitting/flange"
    if f["planarity"] > 0.5 and ext > 1.15:
        return "wall"
    return "other"


def decide(d: dict, g: dict | None, lp: LabelParams) -> dict:
    """Gemini label if confident enough, else the geometry rule (Gemini guess kept)."""
    if g and g.get("label") in VOCAB and float(g.get("confidence", 0)) >= lp.min_conf:
        return {
            "label": g["label"],
            "label_confidence": round(float(g["confidence"]), 2),
            "label_source": "gemini_vision",
            "label_reason": g.get("reason", ""),
        }
    out = {
        "label": label_by_rules(d),
        "label_confidence": lp.rules_conf,
        "label_source": "geometry_rules",
        "label_reason": "fallback: low/no vision confidence",
    }
    if g:
        out["label_vlm"] = {k: g.get(k) for k in ("label", "confidence", "reason")}
    return out


def signature(o: DetectedObject, d: dict) -> str:
    return f"{d['n_points']}|{np.round(o.center, 3).tolist()}"


# ---------- contact sheet ----------


def contact_sheet(
    ids: list[str], details: dict, crops: dict[str, Image.Image], path: Path, lp: LabelParams
) -> None:
    t, cols = lp.thumb_px, lp.sheet_cols
    rows = max((len(ids) + cols - 1) // cols, 1)
    sheet = Image.new("RGB", (cols * t, rows * (t + 28)), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for k, oid in enumerate(ids):
        im, d = crops[oid], details[oid]
        thumb = im.crop((0, 0, im.width // 3, im.height)).resize((t, t))
        x, y = (k % cols) * t, (k // cols) * (t + 28)
        sheet.paste(thumb, (x, y))
        col = (0, 120, 0) if d["label_source"] == "gemini_vision" else (150, 90, 0)
        dr.text((x + 2, y + t + 1), oid, fill=(60, 60, 60), font=font)
        dr.text((x + 2, y + t + 13), f"{d['label'][:20]} {d['label_confidence']:.2f}", fill=col, font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(f"[plot] {path}")


# ---------- main ----------


def ask_gemini(
    todo: list[str],
    details: dict,
    crops: dict[str, Image.Image],
    sig: dict[str, str],
    cache: dict,
    cache_path: Path,
    raw_dir: Path,
    lp: LabelParams,
) -> None:
    """Fill `cache` with Gemini answers for `todo`, saving after every batch."""
    batches = [todo[k : k + lp.batch_size] for k in range(0, len(todo), lp.batch_size)]
    t0 = time.time()

    def job(b: list[str]) -> tuple[list[dict], str]:
        try:
            return gemini_batch(
                batch=[(oid, details[oid], crops[oid]) for oid in b],
                raw_dir=raw_dir,
                lp=lp,
            )
        except Exception as e:  # noqa: BLE001 - one failed batch must not stop the rest
            print(f"  [warn] batch {b[0]}..: {str(e)[:160]}")
            return [], ""

    n_ok = 0
    with ThreadPoolExecutor(lp.workers) as ex:
        for b, (res, model) in zip(batches, ex.map(job, batches), strict=True):
            got = 0
            for r in res:
                rid = str(r.get("id", "")).strip()
                if rid in sig:
                    cache[rid] = {**r, "id": rid, "model": model, "sig": sig[rid]}
                    got += 1
            n_ok += got
            print(f"  batch {b[0]}: {got}/{len(b)} labels ({model or 'failed'})")
            cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    print(f"[gemini] {len(batches)} batches, {n_ok}/{len(todo)} objects labelled in {time.time() - t0:.0f}s")


def run(
    scene_dir: Path,
    report_dir: Path,
    rules_only: bool = False,
    lp: LabelParams = LabelParams(),
) -> list[DetectedObject]:
    scene_dir = Path(scene_dir)
    objects, doc, xyz, rgb, gaussian_labels = load_inputs(scene_dir)
    details = doc["objects"]

    crop_dir = scene_dir / CROPS_DIR
    crop_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(gaussian_labels, kind="stable")
    bounds = np.searchsorted(gaussian_labels[order], np.arange(len(objects) + 1))
    crops: dict[str, Image.Image] = {}
    for k, o in enumerate(objects):
        d = details[o.object_id]
        crops[o.object_id] = object_panels(
            lo=np.asarray(d["bbox_min"]),
            hi=np.asarray(d["bbox_max"]),
            member=order[bounds[k] : bounds[k + 1]],
            xyz=xyz,
            rgb=rgb,
            lp=lp,
        )
        crops[o.object_id].save(crop_dir / f"{o.object_id}.png")
    print(f"[render] {len(crops)} crops -> {crop_dir}")

    cache_path = scene_dir / CACHE_FILE
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    sig = {o.object_id: signature(o=o, d=details[o.object_id]) for o in objects}
    todo = [oid for oid in sig if cache.get(oid, {}).get("sig") != sig[oid]]
    # most informative first (equipment/valves/unnamed, then big pipes) so a quota cut-off hurts least
    todo.sort(key=lambda oid: (details[oid]["hint"] == "pipe segment", -details[oid]["n_points"]))
    if not rules_only and todo:
        ask_gemini(
            todo=todo,
            details=details,
            crops=crops,
            sig=sig,
            cache=cache,
            cache_path=cache_path,
            raw_dir=scene_dir / RAW_DIR,
            lp=lp,
        )

    labelled = []
    for o in objects:
        d = details[o.object_id]
        for key in ("label", "label_confidence", "label_source", "label_reason", "label_vlm"):
            d.pop(key, None)
        hit = cache.get(o.object_id)
        g = hit if hit and hit.get("sig") == sig[o.object_id] and not rules_only else None
        d.update(decide(d=d, g=g, lp=lp))
        labelled.append(o.model_copy(update={"label": d["label"], "confidence": d["label_confidence"]}))
    save_list(scene_dir / OBJECTS_FILE, labelled)
    (scene_dir / DETAILS_FILE).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")

    hist: dict[str, int] = {}
    for d in details.values():
        hist[d["label"]] = hist.get(d["label"], 0) + 1
    n_vlm = sum(d["label_source"] == "gemini_vision" for d in details.values())
    print(f"[labels] {n_vlm}/{len(objects)} from Gemini vision (conf >= {lp.min_conf}), rest geometry rules")
    print("[labels] " + ", ".join(f"{k}: {v}" for k, v in sorted(hist.items(), key=lambda kv: -kv[1])))
    contact_sheet(
        ids=[o.object_id for o in objects],
        details=details,
        crops=crops,
        path=report_dir / "preview" / f"{doc['scene_id']}_label_contact_sheet.png",
        lp=lp,
    )
    return labelled


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--scene-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "equipment_room"
    )
    ap.add_argument("--report-dir", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--rules-only", action="store_true", help="skip Gemini, geometry rules only")
    args = ap.parse_args(argv)
    run(
        scene_dir=args.scene_dir,
        report_dir=args.report_dir,
        rules_only=args.rules_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
