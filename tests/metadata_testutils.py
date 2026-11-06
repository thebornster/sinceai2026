"""Shared helpers for the metadata/diagnostics tests (Dev 3). Not a test module."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from valmet.common.artifacts import OBJECTS_FILE, save_list
from valmet.common.contracts import DetectedObject, OrientedBox
from valmet.metadata.excel_loader import workbook_path
from valmet.metadata.mapping import DETAILS_FILE, METADATA_FIXTURES_DIR

MOCK_METADATA_DIR = METADATA_FIXTURES_DIR / "mock_equipment_room"
MOCK_DETAILS = json.loads((MOCK_METADATA_DIR / DETAILS_FILE).read_text(encoding="utf-8"))
MOCK_OBJECT_IDS = [d["object_id"] for d in MOCK_DETAILS]

requires_workbook = pytest.mark.skipif(
    not workbook_path().exists(),
    reason="equipment workbook not present (data/raw/equipment.xlsx or VALMET_EQUIPMENT_XLSX)",
)

# 25 objects: enough to exhaust healthy pumps/motors and force clones.
BIG_SCENE_LABELS = (
    ["pump"] * 4
    + ["motor"] * 4
    + ["valve"] * 6
    + ["pressure_transmitter"] * 2
    + ["vibration_sensor"] * 2
    + ["pipe"] * 3
    + ["tank"] * 2
    + [None] * 2
)


def isolate_llm(mp: pytest.MonkeyPatch, cache_root: Path) -> None:
    """Never reach the real LLM: blank key, default settings, temp cache, no real client.

    Use from a module-scoped autouse fixture, so it is active before module-scoped
    fixtures that call diagnose_fault(use_llm=True).
    """
    from valmet.diagnostics import llm_client, llm_diagnose

    mp.setenv(llm_client.API_KEY_ENV, "")  # set (even empty) -> .env cannot override it
    for name in llm_client.SETTING_ENVS:
        mp.setenv(name, "")
    mp.setattr(llm_diagnose, "CACHE_ROOT", cache_root)

    def no_real_client(_settings):
        raise AssertionError("tests must never create a real LLM client")

    mp.setattr(llm_client, "make_client", no_real_client)


FAKE_API_KEY = "fake-test-key-0000"


class FakeLLMClient:
    """Stands in for openai.OpenAI. Each reply is a string, or an exception to raise."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
        )


def fake_llm_answer(rules: dict) -> dict:
    """A valid LLM answer built from real rule-based content, so every citation is true."""
    hs = next(e for e in rules["evidence"] if e["field"] == "health_score")
    vib = next(f for f in rules["findings"] if f.get("field") == "vibration_mm_s")
    cause, step = rules["likely_cause"][0], rules["next_steps"][0]
    return {
        "object_id": "WRONG-ID-FROM-MODEL",  # identity must come from the rules
        "tag": rules["tag"],
        "summary": (
            f"{rules['tag']} has health score {hs['value']} ({hs['source']}) and vibration "
            f"{vib['last']} mm/s ({vib['_source']})."
        ),
        "findings": [
            {
                "type": "telemetry_deviation",
                "field": "vibration_mm_s",
                "text": f"Latest vibration {vib['last']} mm/s ({vib['_source']}).",
                "_source": vib["_source"],
                "z_score": 99,  # extra keys are dropped
            }
        ],
        "likely_cause": [
            {
                "text": f"Hypothesis: {cause['text']}",
                "case_id": cause["case_id"],
                "_source": cause["_source"],
            }
        ],
        "explanation": (
            f"Vibration is up at {vib['last']} mm/s ({vib['_source']}). "
            "General background: cavitation makes a pump run rough and lose head."
        ),
        "safety_precautions": [s for s in rules["safety_precautions"] if s["_source"]][:1],
        "next_steps": [
            {  # listed out of order: the parser sorts by priority
                "priority": 2,
                "text": "Listen at the pump casing.",
                "look_for": "Crackling noise.",
                "confirms": "Noise like gravel passing through the pump.",
                "rules_out": "Smooth, steady sound.",
                "basis": "general background",
                "_source": None,
            },
            {
                "priority": 1,
                "text": step["text"],
                "look_for": "Process conditions at the pump.",
                "confirms": "Conditions consistent with cavitation.",
                "rules_out": "Normal process conditions.",
                "basis": "evidence",
                "_source": step["_source"],
            },
        ],
        "evidence": [dict(e) for e in rules["evidence"]],
        "confidence": 0.99,  # must be replaced by the Excel's model_confidence
        "method": "llm",
    }


def wrap_reply(answer: dict) -> str:
    """How Qwen3 tends to answer: a think block, then fenced JSON, then chatter."""
    return (
        "<think>The pump vibration is high {not json}.</think>\n"
        f"```json\n{json.dumps(answer)}\n```\nHope this helps!"
    )


def make_objects(labels: list[str | None], scene_id: str = "test_scene") -> list[DetectedObject]:
    rng = random.Random(0)
    labels = list(labels)
    rng.shuffle(labels)
    out = []
    for i, label in enumerate(labels, start=1):
        center = (rng.uniform(0, 10), rng.uniform(0, 8), rng.uniform(0.3, 2.0))
        out.append(
            DetectedObject(
                object_id=f"obj_{i:04d}",
                scene_id=scene_id,
                label=label,
                confidence=None if label is None else 0.8,
                center=center,
                bbox=OrientedBox(center=center, extent=(0.5, 0.5, 0.5)),
            )
        )
    return out


def write_scene(dir_: Path, objects: list[DetectedObject]) -> Path:
    save_list(Path(dir_) / OBJECTS_FILE, objects)
    return Path(dir_)
