# Valmet Industrial Metaverse

Prototype for the Valmet Automation challenge at Since AI Hackathon 2026. It turns a 3D Gaussian
Splatting scan of an equipment room (`Equipment Room.ply`, ~396 MB) into a navigable digital twin:
detected equipment with bounding boxes, collision-free inspection routes, links to Valmet's
synthetic equipment records, and grounded AI service diagnostics, all shown in a Viser viewer.

The expensive work (scene processing, detection) runs **offline** and writes JSON/NumPy artifacts.
The live viewer only loads those artifacts.

## Architecture

```
                    Equipment Room.ply (data/raw, not in Git)
                                  |
                      preprocessing  --> scene_manifest.json  (world frame: the single source of truth)
                       /                  \
              spatial (Dev 1)          semantic (Dev 2)
   navmap.json + occupancy.npy           objects.json
              paths.json                      |
                       \              metadata (Dev 3) <-- equipment Excel
                        \             asset_mapping.json
                         \                 |
                          \          diagnostics (Dev 3) <-- telemetry / alarms / docs
                           \               |
                            integration -> viewer (Dev 4, Viser)
```

The spatial and semantic pipelines never import each other. They share only `valmet.common`
(contracts, geometry, artifact I/O) and the world frame in `scene_manifest.json`.

## Repository layout

```
data/raw/            real inputs: PLY, Excel (git-ignored)
data/processed/      generated artifacts, one folder per scene_id (git-ignored)
data/fixtures/       small committed fixtures; mock_scene/ = a full mock artifact set
models/weights/      downloaded model weights (git-ignored)
scripts/             CLI entry points (inspect_ply.py, make_mock_fixtures.py)
src/valmet/
  common/            contracts.py (Pydantic schemas), geometry.py (transforms), artifacts.py (I/O)
  preprocessing/     ply_inspect.py; later: Gaussian validation, world-frame estimation
  spatial/           Dev 1
  semantic/          Dev 2
  metadata/          Dev 3 (Excel, mapping)
  diagnostics/       Dev 3 (retrieval, LLM diagnostics)
  integration/       assembles artifacts for the live app
  viewer/            Dev 4
tests/               pytest suite (no GPU, network or real PLY needed)
reports/             small JSON reports (renders/images are git-ignored)
```

## Setup (Python 3.11)

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-core.txt      # or -spatial / -semantic / -viewer
pip install -e .
```

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-core.txt      # or -spatial / -semantic / -viewer
pip install -e .
```

| File | For | Adds to core |
|---|---|---|
| `requirements-core.txt` | everyone, Dev 3 | numpy, scipy, plyfile, pydantic 2, pandas, openpyxl, pytest, ruff |
| `requirements-spatial.txt` | Dev 1 | open3d |
| `requirements-semantic.txt` | Dev 2 | spatial deps, google-genai, pillow, python-dotenv, ultralytics (YOLOE, SAM/SAM 2 wrappers). **Install PyTorch first**, see below |
| `requirements-viewer.txt` | Dev 4 | viser |

Each file includes core via `-r`, so installing more than one is fine.

### PyTorch (semantic only)

Install PyTorch **before** `requirements-semantic.txt`, using the selector at
<https://pytorch.org/get-started/locally/> to match your OS and CUDA driver (check with `nvidia-smi`):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # NVIDIA, CUDA 12.x driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # no GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If you skip this, pip installs the default PyPI torch: CPU-only on Windows/macOS, a CUDA build on Linux.

### SAM 2

- **Recommended:** use the SAM 2 models built into Ultralytics (`from ultralytics import SAM;
  SAM("sam2.1_b.pt")`). No extra install. Weights download on first use, so set the working
  directory or weights path to `models/weights/`.
- **Official `facebookresearch/sam2`:** only if the Ultralytics wrapper isn't enough. It is not on
  PyPI as a stable package and builds from source:
  `git clone https://github.com/facebookresearch/sam2 && pip install -e ./sam2`. It needs a
  matching torch already installed, and may try to compile a CUDA extension (needs the CUDA
  toolkit and, on Windows, MSVC build tools). Set `SAM2_BUILD_CUDA=0` to skip the extension.
  Untested on our machines.

Rendering 2-D views of the splat for detection (e.g. `gsplat`) also needs CUDA compilation. Decide
on it during the semantic investigation; it isn't installed here.

## Commands

```bash
python scripts/inspect_ply.py "data/raw/Equipment Room.ply"          # -> reports/ply_inspection.json
python scripts/inspect_ply.py scene.ply --sample-size 500000 --seed 1 --hash
pytest                                                                # tests
ruff check . && ruff format --check .                                 # lint / format check
ruff check --fix . && ruff format .                                   # auto-fix
python scripts/make_mock_fixtures.py                                  # regenerate mock_scene after schema changes
python scripts/run_viewer.py                                          # viewer on mock_scene (needs requirements-viewer.txt)
python scripts/run_viewer.py --scene-dir data/processed/<scene_id>    # viewer on real artifacts
```

Semantic stage on a preprocessed scene (needs `requirements-semantic.txt`):

```bash
python -m valmet.semantic.segment --scene-dir data/processed/<scene_id>              # objects.json, unlabelled
python -m valmet.semantic.label --scene-dir data/processed/<scene_id> --rules-only   # geometry-rule labels
python -m valmet.semantic.label --scene-dir data/processed/<scene_id>                # Gemini vision, needs GEMINI_API_KEY
python scripts/plan_path.py --target-label pump                                      # route to the best reachable pump
python -m valmet.metadata.mapping --scene-dir data/processed/<scene_id>
VALMET_METADATA_DIR=data/processed/<scene_id> python scripts/run_viewer.py --scene-dir data/processed/<scene_id>
```

The viewer prints its address when it starts. That's normally <http://localhost:8080>, but Viser
moves to the next free port if 8080 is taken. Press `Ctrl+C` to stop it.

The inspector opens the file read-only, memory-maps binary vertex data and reads only a seeded
sample of rows. It reports header, encoding, elements, properties, 3DGS schema conformity (SH
degree), position distributions, non-finite counts, and opacity/scale distributions (raw and
activated, assuming the reference 3DGS encodings: logit opacity, log scale). Exit code 2 means the
file is missing or malformed.

## Shared data contracts (`valmet.common`)

Read `src/valmet/common/contracts.py`; its module docstring is the specification.

- **World frame.** `p_world = SceneManifest.source_to_world @ [p_source, 1]` (4x4, row-major,
  column vectors, translation in the last column). Only preprocessing writes the manifest. No
  pipeline may rescale, recenter or rotate data on its own. The target frame is right-handed,
  +Z up, meters. Until verified, the manifest says `up_axis: null`, `units: "unknown"`,
  `verified: false` and the transform is identity. **We do not yet know the PLY's up axis or scale.**
- **Bounding boxes.** `OrientedBox{center, extent, rotation}`: world center, *full* side lengths, and a
  3x3 rotation whose columns are the box axes in world coordinates (identity = axis-aligned). Same as
  Open3D's `OrientedBoundingBox`.
- **Occupancy grid.** 2-D on world XY (requires `up_axis == "+Z"`). `grid[ix, iy]`, uint8, with
  values from `cell_values` (default free=0, occupied=1, unknown=255). `origin` is the outer corner
  of cell (0,0). Use `world_to_cell` / `cell_to_world`.
- **Models** are frozen, reject unknown fields and NaN/inf, and carry `schema_version: "1.0"`.

| Contract | File (`data/processed/<scene_id>/`) | Producer | Consumers |
|---|---|---|---|
| `SceneManifest` | `scene_manifest.json` | preprocessing | all |
| `DetectedObject` (list) | `objects.json` (+ optional `.npy` Gaussian index files) | semantic | metadata, spatial (targets), viewer |
| `NavigationMapMetadata` | `navmap.json` + `occupancy.npy` | spatial | viewer |
| `NavigationPath` (list) | `paths.json` | spatial | viewer |
| `AssetMapping` (list) | `asset_mapping.json` | metadata | diagnostics, viewer |

```python
from valmet.common.artifacts import MOCK_SCENE_DIR, SceneArtifacts

scene = SceneArtifacts.load(MOCK_SCENE_DIR)  # or data/processed/<scene_id>
assert scene.check_consistency() == []  # cross-file checks: ids, scene_id, grid shape, up axis
```

`data/fixtures/mock_scene/` is a hand-made, internally consistent artifact set (3 objects, 40x30
grid, 1 path, 2 mappings). It isn't derived from the real scan. Develop against it until real
artifacts exist.

## Workstreams

Each developer works in their own package and imports only `valmet.common` from other code.

| Dev | Package | Inputs | Outputs |
|---|---|---|---|
| 1 Spatial | `spatial/` (+ `preprocessing/` for the world frame) | PLY, manifest | manifest (up axis, scale), `navmap.json`, `occupancy.npy`, `paths.json` |
| 2 Semantic | `semantic/` | PLY, manifest | `objects.json`, Gaussian index `.npy` files |
| 3 Metadata/diagnostics | `metadata/`, `diagnostics/` | `objects.json`, equipment Excel, telemetry/docs | `asset_mapping.json`, diagnostic responses |
| 4 Viewer | `viewer/`, `integration/` | PLY + all artifacts | Viser app: splat, boxes, paths, avatar, proximity info, diagnostics panel |

To change a contract, change `contracts.py`, then the tests, then run `make_mock_fixtures.py`, and
tell the team. If the change breaks existing artifacts, bump `SCHEMA_VERSION`.

## Viewer

Pick an object in the Navigate panel and the avatar walks its route to the object's inspection
position. Faulty objects come first in the list. An object is faulty when its `condition_state` in
the equipment data isn't `healthy`.

- `integration/live_scene.py` joins scene artifacts with `valmet.api` once at startup. It's the
  only viewer code that touches other modules.
- Routes come only from `paths.json`, matched on `target_object_id`. The shortest one is used if an
  object has several. Objects with no route show a message instead of moving.
- Play continues the current walk if its object is still selected. Otherwise it starts the selected
  object's route.
- Each route is walked from its own `start`, so the avatar jumps there first.
- Gaussian splats from `viewer_splats.npz` are drawn when present (`--no-splats` skips them).
- Click a box to inspect it: metadata, the semantic crop, "Fly to object" and "Ask copilot", which
  runs `diagnose_fault(use_llm=True)` in the background.
- Device data comes from `valmet.api`, so the viewer ignores `asset_mapping.json` in the scene
  folder. For a real scene, set `$VALMET_METADATA_DIR` to that scene's metadata. Metadata from a
  different `scene_id` is not shown.

## Known limitations

- The viewer doesn't render the Gaussian splat yet, only boxes, routes and the avatar.
- Fault states are read once at startup, so the viewer must restart to pick up changes.
- `data/raw/Equipment Room.ply` was not available when this was set up, so the inspector is only
  validated on synthetic PLYs (binary LE/BE, ASCII). `tests/test_ply_inspect.py::test_real_equipment_room`
  runs automatically once the file is placed in `data/raw/`.
- The up axis, units and scale of the real scan are unknown.
- The inspector cannot sample elements that contain list properties (3DGS files have none).
- Dependency lower bounds were resolved (pip dry-run) on Windows / Python 3.11 only: open3d 0.20,
  viser 1.1, ultralytics 8.4, numpy 2.4. Linux/macOS are expected to work but are untested. SAM 2
  (official) and gsplat are untested.

## Next steps

1. **Dev 1 / shared, first:** run the inspector on the real PLY, commit `reports/ply_inspection.json`,
   then estimate the floor plane and up axis, fix the scale (e.g. from a known door height), and
   write the real `scene_manifest.json`. Everything else depends on this world frame.
2. Dev 1: opacity/scale filtering, floor/obstacle split, occupancy grid, A*.
3. Dev 2: choose the detection route (render views → YOLOE/SAM → lift to 3D, or segment directly in 3D).
4. Dev 3: parse the Excel workbook into typed records and assign mappings to the mock objects.
5. Dev 4: load the splat into the viewer once the real PLY and manifest are ready.
