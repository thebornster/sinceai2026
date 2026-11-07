import json
import shutil

import pytest

from valmet.common.artifacts import CROPS_DIR, OBJECTS_FILE, SEMANTIC_DETAILS_FILE, load_list
from valmet.common.contracts import DetectedObject
from valmet.semantic.label import LabelParams, decide, label_by_rules, parse_list, run

LP = LabelParams()


def detail(hint=None, linearity=0.1, planarity=0.1, size=(0.5, 0.5, 0.5)):
    return {
        "hint": hint,
        "tags": [],
        "n_points": 100,
        "bbox_min": [0.0, 0.0, 0.0],
        "bbox_max": list(size),
        "features": {"linearity": linearity, "planarity": planarity, "height_bottom": 0.0},
    }


def test_confident_gemini_label_wins():
    out = decide(d=detail(), g={"label": "pump", "confidence": 0.8, "reason": "impeller"}, lp=LP)
    assert out == {
        "label": "pump",
        "label_confidence": 0.8,
        "label_source": "gemini_vision",
        "label_reason": "impeller",
    }


@pytest.mark.parametrize(
    "g",
    [
        {"label": "pump", "confidence": LP.min_conf - 0.01},
        {"label": "spaceship", "confidence": 0.99},  # not in the vocabulary
    ],
)
def test_weak_or_unknown_gemini_label_falls_back_and_is_kept(g):
    out = decide(d=detail(hint="fitting/valve"), g=g, lp=LP)
    assert out["label"] == "pipe fitting/flange"
    assert out["label_source"] == "geometry_rules"
    assert out["label_confidence"] == LP.rules_conf
    assert out["label_vlm"]["label"] == g["label"]


def test_no_gemini_answer_uses_rules_without_vlm():
    out = decide(d=detail(), g=None, lp=LP)
    assert out["label"] == "other" and "label_vlm" not in out


@pytest.mark.parametrize(
    "d, expected",
    [
        (detail(hint="pipe segment"), "pipe segment"),
        (detail(linearity=0.9, size=(1.0, 0.1, 0.1)), "pipe segment"),
        (detail(linearity=0.9, size=(0.3, 0.1, 0.1)), "other"),  # too short for a pipe
        (detail(planarity=0.6, size=(2.0, 0.05, 2.0)), "wall"),
        (detail(), "other"),
    ],
)
def test_geometry_rules(d, expected):
    assert label_by_rules(d) == expected


@pytest.mark.parametrize(
    "text",
    [
        '[{"id": "obj_0001", "label": "pump", "confidence": 0.7}]',
        'Sure:\n```json\n[{"id": "obj_0001", "label": "pump", "confidence": 0.7}]\n```',
        '{"labels": [{"id": "obj_0001", "label": "pump", "confidence": 0.7}]}',
    ],
)
def test_parse_reply_variants(text):
    assert parse_list(text) == [{"id": "obj_0001", "label": "pump", "confidence": 0.7}]


def test_rules_only_run_labels_every_object(segmented_scene, tmp_path):
    src, _, _ = segmented_scene
    root = tmp_path / "room"
    shutil.copytree(src, root)

    labelled = run(scene_dir=root, report_dir=tmp_path / "reports", rules_only=True)

    objects = load_list(root / OBJECTS_FILE, DetectedObject)
    assert objects == labelled
    assert all(o.label is not None and o.confidence == LP.rules_conf for o in objects)
    details = json.loads((root / SEMANTIC_DETAILS_FILE).read_text())["objects"]
    assert {d["label_source"] for d in details.values()} == {"geometry_rules"}
    assert "pipe segment" in {o.label for o in objects}
    for o in objects:
        assert (root / CROPS_DIR / f"{o.object_id}.png").exists()
    assert (tmp_path / "reports" / "preview" / "room_label_contact_sheet.png").exists()
