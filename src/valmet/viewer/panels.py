"""Markdown text for the viewer's side panels. No viser needed."""

from __future__ import annotations

from valmet.integration.live_scene import LiveObject


def format_metadata(obj: LiveObject) -> str:
    detected = obj.detected
    confidence = "n/a" if detected.confidence is None else f"{detected.confidence:.0%}"
    lines = [
        f"- **Object:** {detected.object_id} ({detected.label or 'unclassified'})",
        f"- **Detection confidence:** {confidence}",
    ]
    if obj.asset is None:
        return "\n".join([*lines, "- **Equipment:** no record for this object"])

    device, condition = obj.asset["device"], obj.asset["condition"]
    state = condition["condition_state"].replace("_", " ")
    return "\n".join(
        [
            *lines,
            f"- **Equipment:** {device['tag']} ({device['asset_id']})",
            f"- **Area:** {device['area_name']}",
            f"- **Condition:** {state}, health {condition['health_score']}/100",
            f"- **Source:** {condition['_source']}",
        ]
    )


def _section(title: str, items: list[dict]) -> list[str]:
    """A titled bullet list of the items that have text, each citing its Excel row."""
    bullets = [
        f"- {item['text']} [{item['_source']}]" if item.get("_source") else f"- {item['text']}"
        for item in items
        if item.get("text")
    ]
    return [f"**{title}**", "", *bullets, ""] if bullets else []


def format_diagnosis(obj: LiveObject) -> str:
    diagnosis = obj.diagnosis
    if diagnosis is None:
        return "No diagnosis: this object has no equipment record."
    parts = [
        diagnosis["summary"],
        "",
        *_section("Findings", diagnosis["findings"]),
        *_section("Likely cause", diagnosis["likely_cause"]),
        *_section("Next steps", diagnosis["next_steps"]),
        *_section("Safety", diagnosis["safety_precautions"]),
    ]
    return "\n".join(parts).strip()
