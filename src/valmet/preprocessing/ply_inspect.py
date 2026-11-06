"""Read-only PLY inspection: header, 3DGS schema check, sampled statistics.

The file is opened read-only. Binary vertex data is memory-mapped and only a
seeded random sample of rows is touched, so a ~400 MB splat never sits fully in RAM.
No filtering, rotation or rescaling happens here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MAX_HEADER_BYTES = 1 << 20

PLY_TYPES: dict[str, str] = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}  # fmt: skip

GS_REQUIRED = [
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]  # fmt: skip
PERCENTILES = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]


class PlyInspectionError(Exception):
    """Raised for missing, malformed or unsupported PLY files."""


@dataclass
class PlyProperty:
    name: str
    dtype: str  # numpy type code, e.g. "f4"
    list_count_dtype: str | None = None  # set for list properties


@dataclass
class PlyElement:
    name: str
    count: int
    properties: list[PlyProperty] = field(default_factory=list)

    @property
    def has_lists(self) -> bool:
        return any(p.list_count_dtype for p in self.properties)


@dataclass
class PlyHeader:
    format: str  # ascii | binary_little_endian | binary_big_endian
    version: str
    elements: list[PlyElement]
    header_bytes: int
    comments: list[str]

    def element(self, name: str) -> PlyElement | None:
        return next((e for e in self.elements if e.name == name), None)


def read_header(path: Path) -> PlyHeader:
    path = Path(path)
    if not path.exists():
        raise PlyInspectionError(f"file not found: {path}")
    if not path.is_file():
        raise PlyInspectionError(f"not a regular file: {path}")

    with path.open("rb") as f:
        raw = f.read(MAX_HEADER_BYTES)
    if not raw.startswith(b"ply"):
        raise PlyInspectionError(f"not a PLY file (missing 'ply' magic): {path}")
    m = re.search(rb"end_header\r?\n", raw)
    if m is None:
        raise PlyInspectionError("no 'end_header' line within the first 1 MiB")
    header_bytes = m.end()
    lines = raw[:header_bytes].decode("ascii", errors="strict").splitlines()

    fmt = version = None
    elements: list[PlyElement] = []
    comments: list[str] = []
    for lineno, line in enumerate(lines[1:-1], start=2):
        tok = line.split()
        if not tok:
            continue
        kw = tok[0]
        try:
            if kw == "format":
                fmt, version = tok[1], tok[2]
                if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
                    raise PlyInspectionError(f"unknown PLY format {fmt!r}")
            elif kw in ("comment", "obj_info"):
                comments.append(line.partition(" ")[2])
            elif kw == "element":
                count = int(tok[2])
                if count < 0:
                    raise ValueError("negative element count")
                elements.append(PlyElement(tok[1], count))
            elif kw == "property":
                if not elements:
                    raise PlyInspectionError(f"line {lineno}: property before any element")
                if tok[1] == "list":
                    elements[-1].properties.append(
                        PlyProperty(tok[4], PLY_TYPES[tok[3]], list_count_dtype=PLY_TYPES[tok[2]])
                    )
                else:
                    elements[-1].properties.append(PlyProperty(tok[2], PLY_TYPES[tok[1]]))
            else:
                raise PlyInspectionError(f"line {lineno}: unexpected header keyword {kw!r}")
        except (IndexError, ValueError, KeyError) as exc:
            raise PlyInspectionError(f"line {lineno}: malformed header line {line!r}") from exc

    if fmt is None:
        raise PlyInspectionError("header has no 'format' line")
    if not elements:
        raise PlyInspectionError("header declares no elements")
    return PlyHeader(fmt, version or "", elements, header_bytes, comments)


def _struct_dtype(el: PlyElement, fmt: str) -> np.dtype:
    order = "<" if fmt == "binary_little_endian" else ">"
    return np.dtype([(p.name, order + p.dtype) for p in el.properties])


def gaussian_schema(names: list[str]) -> dict[str, Any]:
    """Compare vertex property names to the reference 3DGS (Kerbl et al. 2023) layout."""
    present = set(names)
    missing = [n for n in GS_REQUIRED if n not in present]
    n_rest = sum(1 for n in names if re.fullmatch(r"f_rest_\d+", n))
    sh_degree = None
    for d in range(5):
        if n_rest == 3 * ((d + 1) ** 2 - 1):
            sh_degree = d
    known = set(GS_REQUIRED) | {"nx", "ny", "nz"}
    extra = [n for n in names if n not in known and not re.fullmatch(r"f_rest_\d+", n)]
    return {
        "is_conventional_3dgs": not missing and sh_degree is not None,
        "missing_required": missing,
        "f_rest_count": n_rest,
        "sh_degree": sh_degree,
        "has_normals": {"nx", "ny", "nz"} <= present,
        "extra_properties": extra,
    }


def _stats(values: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    out: dict[str, Any] = {"n": int(v.size), "nonfinite": int((~finite).sum())}
    v = v[finite]
    if v.size == 0:
        return out
    out.update(
        min=float(v.min()),
        max=float(v.max()),
        mean=float(v.mean()),
        std=float(v.std()),
        percentiles={
            str(p): float(q)
            for p, q in zip(PERCENTILES, np.percentile(v, PERCENTILES), strict=True)
        },
    )
    return out


def _sample_rows(path: Path, header: PlyHeader, el: PlyElement, n: int, seed: int) -> np.ndarray:
    """Seeded sample of `n` rows of `el` (sorted indices), without reading the whole file."""
    rng = np.random.default_rng(seed)
    n = min(n, el.count)
    idx = np.sort(rng.choice(el.count, size=n, replace=False)) if n < el.count else np.arange(n)

    if header.format == "ascii":
        return _sample_ascii(path, header, el, idx)

    # Binary: elements before `el` must be fixed-size to compute its offset.
    offset = header.header_bytes
    for prior in header.elements:
        if prior is el:
            break
        if prior.has_lists:
            raise PlyInspectionError(
                f"cannot seek past list-valued element {prior.name!r} before {el.name!r}"
            )
        offset += prior.count * _struct_dtype(prior, header.format).itemsize
    if el.has_lists:
        raise PlyInspectionError(f"element {el.name!r} has list properties; sampling unsupported")

    dtype = _struct_dtype(el, header.format)
    needed = offset + el.count * dtype.itemsize
    size = path.stat().st_size
    if size < needed:
        raise PlyInspectionError(
            f"file truncated: header implies >= {needed} bytes, file has {size}"
        )
    mm = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(el.count,))
    try:
        return np.array(mm[idx])
    finally:
        del mm


def _sample_ascii(path: Path, header: PlyHeader, el: PlyElement, idx: np.ndarray) -> np.ndarray:
    if el.has_lists or header.elements[0] is not el:
        raise PlyInspectionError("ASCII sampling supports only a list-free first element")
    dtype = np.dtype([(p.name, p.dtype) for p in el.properties])
    rows = np.empty(len(idx), dtype=dtype)
    wanted = iter(enumerate(idx.tolist()))
    k, target = next(wanted, (None, None))
    with path.open("rb") as f:
        f.seek(header.header_bytes)
        for i in range(el.count):
            line = f.readline()
            if not line:
                raise PlyInspectionError(f"file truncated: {el.name} row {i} of {el.count} missing")
            if i == target:
                vals = line.split()
                if len(vals) != len(el.properties):
                    raise PlyInspectionError(
                        f"{el.name} row {i}: expected {len(el.properties)} values"
                    )
                rows[k] = tuple(float(x) for x in vals)
                k, target = next(wanted, (None, None))
                if target is None:
                    break
    return rows


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def inspect_ply(
    path: Path, sample_size: int = 200_000, seed: int = 0, compute_hash: bool = False
) -> dict[str, Any]:
    path = Path(path)
    if sample_size <= 0:
        raise PlyInspectionError("sample_size must be positive")
    header = read_header(path)
    report: dict[str, Any] = {
        "file": {"path": str(path), "name": path.name, "size_bytes": path.stat().st_size},
        "header": {
            "format": header.format,
            "version": header.version,
            "header_bytes": header.header_bytes,
            "comments": header.comments,
            "elements": [
                {
                    "name": e.name,
                    "count": e.count,
                    "properties": [
                        {"name": p.name, "type": p.dtype, "list": p.list_count_dtype is not None}
                        for p in e.properties
                    ],
                }
                for e in header.elements
            ],
        },
        "warnings": [],
    }
    if compute_hash:
        report["file"]["sha256"] = _sha256(path)

    vertex = header.element("vertex")
    if vertex is None:
        raise PlyInspectionError("no 'vertex' element in PLY")
    names = [p.name for p in vertex.properties]
    if not {"x", "y", "z"} <= set(names):
        raise PlyInspectionError("vertex element lacks x/y/z properties")

    if header.format != "ascii" and not any(e.has_lists for e in header.elements):
        expected = header.header_bytes + sum(
            e.count * _struct_dtype(e, header.format).itemsize for e in header.elements
        )
        report["file"]["expected_size_bytes"] = expected
        if report["file"]["size_bytes"] > expected:
            report["warnings"].append(
                f"{report['file']['size_bytes'] - expected} trailing bytes after declared data"
            )

    schema = gaussian_schema(names)
    report["gaussian_schema"] = schema
    if not schema["is_conventional_3dgs"]:
        report["warnings"].append("vertex properties do not match the conventional 3DGS layout")

    report["sample"] = {"requested": sample_size, "seed": seed, "vertex_count": vertex.count}
    if vertex.count == 0:
        report["warnings"].append("vertex element is empty")
        return report

    rows = _sample_rows(path, header, vertex, sample_size, seed)
    report["sample"]["size"] = int(len(rows))
    xyz = np.stack([rows["x"], rows["y"], rows["z"]], axis=1).astype(np.float64)
    finite_rows = np.isfinite(xyz).all(axis=1)
    report["positions"] = {
        "nonfinite_rows": int((~finite_rows).sum()),
        "axes": {a: _stats(xyz[:, i]) for i, a in enumerate("xyz")},
    }
    if finite_rows.any():
        lo = np.percentile(xyz[finite_rows], 1, axis=0)
        hi = np.percentile(xyz[finite_rows], 99, axis=0)
        report["positions"]["p1_p99_extent"] = (hi - lo).tolist()
        report["positions"]["p1_p99_center"] = ((hi + lo) / 2).tolist()

    if "opacity" in names:
        raw = rows["opacity"].astype(np.float64)
        act = 1.0 / (1.0 + np.exp(-np.clip(raw, -60, 60)))
        report["opacity"] = {
            "assumed_encoding": "logit (reference 3DGS); activated = sigmoid(raw)",
            "raw": _stats(raw),
            "activated": _stats(act),
            "fraction_activated_below": {
                str(t): float(np.mean(act[np.isfinite(act)] < t)) for t in (0.005, 0.05, 0.1, 0.5)
            },
        }

    scale_names = [n for n in ("scale_0", "scale_1", "scale_2") if n in names]
    if len(scale_names) == 3:
        raw = np.stack([rows[n] for n in scale_names], axis=1).astype(np.float64)
        act = np.exp(np.clip(raw, -60, 60))
        report["scale"] = {
            "assumed_encoding": "log (reference 3DGS); activated = exp(raw), source units",
            "raw": {n: _stats(raw[:, i]) for i, n in enumerate(scale_names)},
            "activated_max_axis": _stats(act.max(axis=1)),
            "activated_min_axis": _stats(act.min(axis=1)),
        }

    rot_names = [f"rot_{i}" for i in range(4)]
    if all(n in names for n in rot_names):
        q = np.stack([rows[n] for n in rot_names], axis=1).astype(np.float64)
        report["rotation_quaternion_norm"] = _stats(np.linalg.norm(q, axis=1))

    if report["positions"]["nonfinite_rows"]:
        report["warnings"].append("non-finite vertex positions present in sample")
    return report
