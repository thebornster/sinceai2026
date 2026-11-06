"""Rule-based fault diagnosis built only from Excel evidence.

Every statement points at the Excel row it came from (``_source``), and
``evidence`` lists every raw cell value used, as {source, field, value}.
Derived numbers (peer means, z-scores, counts) are labelled as derived and are
not in ``evidence``.
"""

from __future__ import annotations

import math
import statistics
from functools import cache
from typing import Any

from valmet.diagnostics.evidence import _json, get_evidence, resolve
from valmet.metadata.registry import get_registry

LOTO = "Follow site lock-out/tag-out procedures before any physical inspection."
N_TOP_FINDINGS = 3


# --------------------------------------------------------------------------- #
# Healthy-peer baseline
# --------------------------------------------------------------------------- #
@cache
def _peer_stats(asset_class: str, exclude_asset: str, field: str) -> dict[str, Any] | None:
    """Mean/std of `field` over all telemetry rows of healthy devices of this class."""
    peers = [
        r
        for r in get_registry().by_class(asset_class)
        if r.is_healthy and r.asset_id != exclude_asset
    ]
    values = []
    for p in peers:
        for t in p.telemetry:
            v = t.model_dump().get(field)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                values.append(v)
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "n": len(values),
        "assets": sorted(p.asset_id for p in peers),
    }


def _telemetry_findings(ev: dict[str, Any]) -> list[dict[str, Any]]:
    asset_class, asset_id = ev["device"]["asset_class"], ev["device"]["asset_id"]
    scored = []
    for field, s in ev["telemetry"].items():
        peer = _peer_stats(asset_class, asset_id, field)
        if peer is None:
            continue
        diff = s["last"] - peer["mean"]
        if diff == 0:
            continue
        scale = max(peer["std"], 1e-6 * max(1.0, abs(peer["mean"])))
        z = diff / scale
        direction = "higher" if diff > 0 else "lower"
        scored.append(
            {
                "type": "telemetry_deviation",
                "field": field,
                "direction": direction,
                "last": s["last"],
                "last_at": s["last_at"],
                "_source": s["_sources"]["last"],
                "derived": {
                    "healthy_peer_mean": round(peer["mean"], 4),
                    "healthy_peer_std": round(peer["std"], 4),
                    "z_score": round(z, 2),
                    "peer_assets": peer["assets"],
                    "peer_rows": peer["n"],
                },
                "text": (
                    f"{field} latest value {s['last']} is {direction} than the healthy "
                    f"{asset_class} mean of {peer['mean']:.3g} (z = {z:+.1f})."
                ),
            }
        )
    scored.sort(key=lambda f: -abs(f["derived"]["z_score"]))
    return scored[:N_TOP_FINDINGS]


def _alarm_findings(object_id: str, ev: dict[str, Any]) -> list[dict[str, Any]]:
    _, rec = resolve(object_id)
    out: list[dict[str, Any]] = [
        {
            "type": "alarm_summary",
            "total": ev["alarms"]["total"],
            "by_severity": ev["alarms"]["by_severity"],
            "text": f"{ev['alarms']['total']} alarms/events in the sample: "
            + ", ".join(f"{n} {sev}" for sev, n in ev["alarms"]["by_severity"].items())
            + ".",
        }
    ]
    newest_high = next(
        (a for a in rec.alarms if a.model_dump().get("severity") == "high"), None
    )  # newest first
    if newest_high is not None:
        d = newest_high.model_dump()
        out.append(
            {
                "type": "latest_high_alarm",
                **{
                    k: _json(d.get(k))
                    for k in (
                        "timestamp",
                        "event_type",
                        "severity",
                        "event_code",
                        "description",
                        "status",
                    )
                },
                "_source": newest_high.source_ref,
                "text": (
                    f"Newest high-severity event: {d.get('event_type')} / "
                    f"{d.get('event_code')} at {_json(d.get('timestamp'))} ({d.get('status')})."
                ),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _dedupe(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen, out = set(), []
    for it in items:
        k = " ".join(str(it[key]).lower().split())
        if it[key] and k not in seen:
            seen.add(k)
            out.append(it)
    return out


def _pretty(code: str) -> str:
    return str(code).replace("_", " ").lower()


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
def diagnose_fault(object_id: str, use_llm: bool = False) -> dict[str, Any]:
    result = _rule_based(object_id)
    if use_llm:
        # LLM path not implemented yet: never crash the viewer, return the rules result.
        result["method"] = "rule_based_fallback"
    return result


def _rule_based(object_id: str) -> dict[str, Any]:
    ev = get_evidence(object_id)
    dev, cond = ev["device"], ev["condition"]
    evidence: list[dict[str, Any]] = []

    def cite(source: str, field: str, value: Any) -> None:
        item = {"source": source, "field": field, "value": value}
        if item not in evidence:
            evidence.append(item)

    # Summary
    summary = (
        f"{dev['tag']} ({_pretty(dev['asset_class'])}, {dev['area_name']}) is "
        f"{_pretty(cond['condition_state'])} with health score {cond['health_score']}/100 "
        f"and {cond['urgency']} urgency. Current state: {cond['issue_summary']} "
        f"({cond['issue_code']})."
    )
    for f in ("tag", "asset_class", "area_name"):
        cite(dev["_source"], f, dev[f])
    for f in ("condition_state", "health_score", "urgency", "issue_summary", "issue_code"):
        cite(cond["_source"], f, cond[f])

    # Findings
    findings = _telemetry_findings(ev) + _alarm_findings(object_id, ev)
    for f in findings:
        if f["type"] == "telemetry_deviation":
            cite(f["_source"], f["field"], f["last"])
        elif f["type"] == "latest_high_alarm":
            for k in ("timestamp", "event_type", "severity", "event_code"):
                cite(f["_source"], k, f[k])

    # Likely cause / safety / next steps from similar cases
    cases = ev["similar_cases"]
    likely_cause = _dedupe(
        [
            {
                "text": c["historical_interpretation"],
                "case_id": c["case_id"],
                "_source": c["_source"],
            }
            for c in cases
        ],
        "text",
    )
    safety = _dedupe(
        [
            {"text": c["safe_evidence_workflow"], "case_id": c["case_id"], "_source": c["_source"]}
            for c in cases
        ],
        "text",
    ) + [{"text": LOTO, "case_id": None, "_source": None}]
    next_steps = _dedupe(
        [{"text": cond["recommended_next_check"], "_source": cond["_source"]}]
        + [{"text": c["recommended_next_check"], "_source": c["_source"]} for c in cases],
        "text",
    )
    for c in likely_cause:
        cite(c["_source"], "historical_interpretation", c["text"])
    for s in safety:
        if s["_source"]:
            cite(s["_source"], "safe_evidence_workflow", s["text"])
    for n in next_steps:
        cite(n["_source"], "recommended_next_check", n["text"])
    cite(cond["_source"], "model_confidence", cond["model_confidence"])

    return {
        "object_id": object_id,
        "device_id": ev["object"]["device_id"],
        "asset_id": dev["asset_id"],
        "tag": dev["tag"],
        "is_target": ev["object"]["is_target"],
        "summary": summary,
        "findings": findings,
        "likely_cause": likely_cause,
        "safety_precautions": safety,
        "next_steps": next_steps,
        "evidence": evidence,
        "confidence": cond["model_confidence"],
        "method": "rule_based",
    }
