import json
import threading
import time
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from metadata_testutils import (
    FAKE_API_KEY,
    MOCK_OBJECT_IDS,
    FakeLLMClient,
    fake_llm_answer,
    isolate_llm,
    requires_workbook,
    wrap_reply,
)
from valmet import api
from valmet.diagnostics import llm_client, llm_diagnose
from valmet.diagnostics.citations import check_citations
from valmet.diagnostics.llm_diagnose import LOTO
from valmet.metadata.excel_loader import load_workbook
from valmet.metadata.mapping import METADATA_DIR_ENV

pytestmark = requires_workbook
ALL_OBJECTS = MOCK_OBJECT_IDS


@pytest.fixture(scope="module", autouse=True)
def _no_real_llm(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        isolate_llm(mp, tmp_path_factory.mktemp("llm_cache"))
        yield


@pytest.fixture(autouse=True)
def _mock_scene(monkeypatch):
    monkeypatch.delenv(METADATA_DIR_ENV, raising=False)  # always test against the mock


@pytest.fixture(scope="module")
def rows_by_source():
    out = {}
    for df in load_workbook().values():
        for row in df.to_dict("records"):
            out[row["_source"]] = row
    return out


def _norm(v):
    if hasattr(v, "to_pydatetime"):
        v = v.to_pydatetime()
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if hasattr(v, "item"):
        return v.item()
    return v


# --------------------------------------------------------------------------- #
# diagnose_fault
# --------------------------------------------------------------------------- #
def test_diagnose_obj_0001_mentions_p002_and_cavitation():
    d = api.diagnose_fault("obj_0001")
    assert "P-002" in d["summary"]
    assert "CAVITATION_RISK" in d["summary"]
    assert d["tag"] == "P-002" and d["asset_id"] == "ASSET-014" and d["is_target"]
    assert d["method"] == "rule_based"
    assert d["confidence"] == 0.68


def test_diagnose_obj_0001_findings_include_vibration():
    d = api.diagnose_fault("obj_0001")
    tele = [f for f in d["findings"] if f["type"] == "telemetry_deviation"]
    assert 1 <= len(tele) <= 3
    assert "vibration_mm_s" in [f["field"] for f in tele]
    vib = next(f for f in tele if f["field"] == "vibration_mm_s")
    assert vib["direction"] == "higher"
    # Ranked by |z| descending.
    zs = [abs(f["derived"]["z_score"]) for f in tele]
    assert zs == sorted(zs, reverse=True)
    types = [f["type"] for f in d["findings"]]
    assert "alarm_summary" in types and "latest_high_alarm" in types


@pytest.mark.parametrize("object_id", ALL_OBJECTS)
def test_every_evidence_source_exists_and_value_matches(object_id, rows_by_source):
    d = api.diagnose_fault(object_id)
    assert d["evidence"]
    for e in d["evidence"]:
        assert e["source"] in rows_by_source, e
        assert _norm(rows_by_source[e["source"]][e["field"]]) == e["value"], e


@pytest.mark.parametrize("object_id", ALL_OBJECTS)
def test_every_cited_source_in_output_is_in_evidence(object_id):
    d = api.diagnose_fault(object_id)
    ev_sources = {e["source"] for e in d["evidence"]}
    for section in ("findings", "likely_cause", "safety_precautions", "next_steps"):
        for item in d[section]:
            if item.get("_source"):
                assert item["_source"] in ev_sources, (section, item)


def _alarm(d):
    return next(f for f in d["findings"] if f["type"] == "latest_high_alarm")


def test_latest_high_alarm_skips_routine_events():
    a = _alarm(api.diagnose_fault("obj_0001"))
    # R461 (SERVICE_RECOMMENDATION / NORMAL_CHECK) is newer but routine.
    assert a["_source"] == "Alarms_Events!R702"
    assert (a["event_type"], a["event_code"], a["severity"]) == (
        "DIAGNOSTIC_ALERT",
        "CAVITATION_RISK",
        "high",
    )
    assert a["text"].startswith("Newest high-severity fault alarm:")


@pytest.mark.parametrize("object_id,n_high", [("obj_0002", 7), ("obj_0003", 12)])
def test_no_fault_alarm_is_stated_with_the_same_shape(object_id, n_high, rules_0001):
    d = api.diagnose_fault(object_id)
    a = _alarm(d)
    assert set(a) == set(_alarm(rules_0001))  # same keys as when an alarm exists
    assert a["_source"] is None and a["event_type"] is None
    assert a["text"] == (
        f"No high-severity fault alarm in the sample; all {n_high} high-severity events "
        "are routine records."
    )
    summary = next(f for f in d["findings"] if f["type"] == "alarm_summary")
    assert summary["text"].endswith("none are fault alarms, all are routine records.")


@pytest.mark.parametrize("object_id", ALL_OBJECTS)
def test_no_routine_event_cited_as_an_alarm(object_id):
    from valmet.diagnostics.evidence import ROUTINE_EVENT_CODES, ROUTINE_EVENT_TYPES

    a = _alarm(api.diagnose_fault(object_id))
    assert a["event_type"] not in ROUTINE_EVENT_TYPES
    assert a["event_code"] not in ROUTINE_EVENT_CODES


def test_safety_includes_loto_and_next_steps_deduped():
    d = api.diagnose_fault("obj_0001")
    assert any("lock-out/tag-out" in s["text"] for s in d["safety_precautions"])
    texts = [n["text"].lower().strip() for n in d["next_steps"]]
    assert len(texts) == len(set(texts))
    assert d["next_steps"][0]["_source"] == "Current_State!R15"
    assert d["likely_cause"]


@pytest.mark.parametrize("object_id", ["obj_0001", "obj_0002"])
def test_use_llm_without_key_falls_back_to_rules(object_id):
    rules = api.diagnose_fault(object_id)
    llm = api.diagnose_fault(object_id, use_llm=True)
    assert llm["method"] == "rule_based_fallback"
    assert "FEATHERLESS_API_KEY" in llm["fallback_reason"]
    assert rules["method"] == "rule_based"
    drop = ("method", "fallback_reason")
    assert {k: v for k, v in llm.items() if k not in drop} == {
        k: v for k, v in rules.items() if k != "method"
    }
    json.dumps(llm)


def test_use_llm_unknown_object_still_raises():
    with pytest.raises(KeyError, match="obj_9999"):
        api.diagnose_fault("obj_9999", use_llm=True)


# --------------------------------------------------------------------------- #
# diagnose_fault(use_llm=True) with a fake client (never the real API)
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_llm(monkeypatch, tmp_path):
    """install(*replies) -> FakeLLMClient used for every LLM call in this test."""
    monkeypatch.setenv(llm_client.API_KEY_ENV, FAKE_API_KEY)
    monkeypatch.setattr(llm_diagnose, "CACHE_ROOT", tmp_path)

    def install(*replies):
        client = FakeLLMClient(replies)
        monkeypatch.setattr(llm_client, "make_client", lambda _settings: client)
        return client

    return install


@pytest.fixture(scope="module")
def rules_0001():
    return api.diagnose_fault("obj_0001")


def _cache_file(tmp_path):
    return tmp_path / "mock_equipment_room" / "llm_diagnoses" / "obj_0001.json"


def test_llm_reply_with_think_and_fences_parses(fake_llm, rules_0001, tmp_path):
    client = fake_llm(wrap_reply(fake_llm_answer(rules_0001)))
    d = api.diagnose_fault("obj_0001", use_llm=True)

    assert d["method"] == "llm"
    assert d["citation_check"]["ok"], d["citation_check"]
    assert d["citation_check"]["failed"] == []
    assert d["citation_check"]["checked"] >= len(rules_0001["evidence"])
    # Identity and confidence come from the rules/Excel, not from the model.
    for k in ("object_id", "device_id", "asset_id", "tag", "is_target"):
        assert d[k] == rules_0001[k]
    assert d["confidence"] == rules_0001["confidence"] == 0.68
    assert d["evidence"] == rules_0001["evidence"]
    assert "z_score" not in d["findings"][0]  # extra keys dropped by the schema
    assert any("lock-out/tag-out" in s["text"] for s in d["safety_precautions"])
    # Settings and run metadata are recorded.
    rec = d["llm"]
    assert rec["model"] == rec["response_model"] == "Qwen/Qwen3.8-Flash-Next"
    assert rec["role"] == "live"
    assert (rec["temperature"], rec["top_p"], rec["enable_thinking"]) == (0.7, 0.8, False)
    assert rec["base_url"] == "https://api.featherless.ai/v1"
    assert rec["attempts"] == 1 and rec["created_at"].endswith("+00:00")
    # The request used the Qwen3 defaults from the smoke test.
    (call,) = client.calls
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert (call["temperature"], call["top_p"]) == (0.7, 0.8)
    system, user = call["messages"]
    for rule in (
        "must come from the user message",
        'labelled "derived from <sources>"',
        '"General background:"',
        'never "alarms"',
        '"Hypothesis:"',
        '"rules_out"',
        "never autonomous control actions",
        "SERVICE_RECOMMENDATION",
        "NORMAL_CHECK",
    ):
        assert rule in system["content"], rule
    assert "Current_State!R15" in user["content"] and "Alarms_Events!R" in user["content"]
    # LLM-only additions: explanation and a prioritised, sorted inspection sequence.
    assert d["explanation"].startswith("Vibration is up")
    assert [s["priority"] for s in d["next_steps"]] == [1, 2]
    assert set(d["next_steps"][0]) >= {"look_for", "confirms", "rules_out", "basis"}
    assert rec["retry_reason"] is None and rec["timeout_s"] == 60
    # The key never ends up in the result or the cache.
    assert FAKE_API_KEY not in json.dumps(d)
    assert FAKE_API_KEY not in _cache_file(tmp_path).read_text(encoding="utf-8")


def _wrong_health_score(rules):
    answer = fake_llm_answer(rules)
    next(e for e in answer["evidence"] if e["field"] == "health_score")["value"] = 99  # Excel: 45
    return answer


def test_wrong_value_is_caught_and_removed(fake_llm, rules_0001):
    bad = wrap_reply(_wrong_health_score(rules_0001))
    client = fake_llm(bad, bad)  # the repair repeats the mistake
    d = api.diagnose_fault("obj_0001", use_llm=True)

    report = d["citation_check"]
    assert d["method"] == "llm" and not report["ok"]
    assert [f["kind"] for f in report["failed"]] == ["value_mismatch"]
    (fail,) = report["failed"]
    assert (fail["source"], fail["field"], fail["value"], fail["expected"]) == (
        "Current_State!R15",
        "health_score",
        99,
        45,
    )
    assert not any(e["field"] == "health_score" for e in d["evidence"])
    assert len(d["evidence"]) == len(rules_0001["evidence"]) - 1
    repair = d["llm"]["citation_repair"]
    assert repair["failed_before"] == repair["failed_after"] == report["failed"]
    assert d["llm"]["attempts"] == 2 and len(client.calls) == 2


def test_citation_failure_is_repaired_with_the_exact_failures(fake_llm, rules_0001):
    client = fake_llm(
        wrap_reply(_wrong_health_score(rules_0001)), wrap_reply(fake_llm_answer(rules_0001))
    )
    d = api.diagnose_fault("obj_0001", use_llm=True)
    assert d["citation_check"]["ok"] and d["citation_check"]["failed"] == []
    repair = d["llm"]["citation_repair"]
    assert repair["used"] is True and repair["failed_after"] == []
    assert [f["kind"] for f in repair["failed_before"]] == ["value_mismatch"]
    assert d["llm"]["attempts"] == 2
    ask = client.calls[1]["messages"]
    assert [m["role"] for m in ask] == ["system", "user", "assistant", "user"]
    assert "failed the citation check" in ask[-1]["content"]
    assert '"expected": 45' in ask[-1]["content"] and "Current_State!R15" in ask[-1]["content"]
    assert ask[2]["content"].startswith("{")  # our parsed JSON, not the raw <think> text


def test_citation_repair_error_keeps_the_first_answer(fake_llm, rules_0001):
    fake_llm(wrap_reply(_wrong_health_score(rules_0001)), TimeoutError("slow"))
    d = api.diagnose_fault("obj_0001", use_llm=True)
    assert d["method"] == "llm" and not d["citation_check"]["ok"]
    repair = d["llm"]["citation_repair"]
    assert repair["used"] is False and repair["error"] == "timeout after 60 s"
    assert d["llm"]["attempts"] == 2


def test_no_citation_repair_when_the_check_passes(fake_llm, rules_0001):
    client = fake_llm(wrap_reply(fake_llm_answer(rules_0001)))
    d = api.diagnose_fault("obj_0001", use_llm=True)
    assert d["llm"]["citation_repair"] is None and len(client.calls) == 1


def test_reference_missing_from_evidence_is_flagged(fake_llm, rules_0001):
    answer = fake_llm_answer(rules_0001)
    answer["summary"] += " See also Alarms_Events!R2."  # cited but not in evidence
    fake_llm(wrap_reply(answer), wrap_reply(answer))
    report = api.diagnose_fault("obj_0001", use_llm=True)["citation_check"]
    assert not report["ok"]
    assert report["failed"] == [
        {"kind": "uncited_reference", "source": "Alarms_Events!R2", "where": "summary"}
    ]


# --- code-side filters ------------------------------------------------------ #
def test_safety_filter_keeps_case_workflow_and_drops_invented_items(fake_llm, rules_0001):
    answer = fake_llm_answer(rules_0001)
    workflow = next(s for s in rules_0001["safety_precautions"] if s["_source"])
    answer["safety_precautions"] = [
        {"text": "Wear PPE for chemical exposure.", "case_id": None, "_source": None},
        {"text": f"{workflow['text']} ({workflow['_source']})", "case_id": "x", "_source": None},
        {"text": "Lock out/tag out. Verify the pump is depressurised.", "_source": None},
        {"text": LOTO, "case_id": None, "_source": None},
    ]
    answer["evidence"] = [e for e in answer["evidence"] if e["field"] != "safe_evidence_workflow"]
    fake_llm(wrap_reply(answer))
    d = api.diagnose_fault("obj_0001", use_llm=True)

    texts = [s["text"] for s in d["safety_precautions"]]
    assert texts == [f"{workflow['text']} ({workflow['_source']})", LOTO]
    assert d["safety_precautions"][0]["_source"] == workflow["_source"]
    assert d["safety_precautions"][0]["case_id"] == workflow["case_id"]
    filters = d["llm"]["filters"]
    assert filters["safety_dropped"] == [
        "Wear PPE for chemical exposure.",
        "Lock out/tag out. Verify the pump is depressurised.",
    ]
    assert filters["safety_added"] == []
    # The kept workflow's row is cited in evidence by the filter, so the check passes.
    cited = {"source": workflow["_source"], "field": "safe_evidence_workflow"}
    assert {**cited, "value": workflow["text"]} in d["evidence"]
    assert d["citation_check"]["ok"], d["citation_check"]["failed"]


def test_safety_filter_adds_the_case_workflow_when_missing(fake_llm, rules_0001):
    answer = fake_llm_answer(rules_0001)
    answer["safety_precautions"] = [{"text": "Be careful.", "case_id": None, "_source": None}]
    fake_llm(wrap_reply(answer))
    d = api.diagnose_fault("obj_0001", use_llm=True)
    workflow = next(s for s in rules_0001["safety_precautions"] if s["_source"])
    assert [s["text"] for s in d["safety_precautions"]] == [workflow["text"], LOTO]
    assert d["llm"]["filters"]["safety_dropped"] == ["Be careful."]
    assert d["llm"]["filters"]["safety_added"] == [workflow["text"]]


def test_unsupported_evidence_step_is_downgraded_to_hypothesis(fake_llm, rules_0001):
    answer = fake_llm_answer(rules_0001)
    answer["next_steps"].append(
        {
            "priority": 3,
            "text": "Check the suction strainer for clogging.",  # not in Current_State!R15
            "look_for": "Debris.",
            "confirms": "Blocked strainer.",
            "rules_out": "Clean strainer.",
            "basis": "evidence",
            "_source": "Current_State!R15",
        }
    )
    fake_llm(wrap_reply(answer))
    d = api.diagnose_fault("obj_0001", use_llm=True)
    basis = {s["priority"]: s["basis"] for s in d["next_steps"]}
    assert basis == {1: "evidence", 2: "general background", 3: "hypothesis"}
    assert d["llm"]["filters"]["basis_downgraded"] == [
        {
            "priority": 3,
            "text": "Check the suction strainer for clogging.",
            "reason": "Current_State!R15 does not contain the step's instruction",
        }
    ]


def test_basis_filter_ignores_filler_words_and_punctuation(fake_llm, rules_0001):
    """'Inspect the process conditions' is supported by 'inspect process conditions'."""
    answer = fake_llm_answer(rules_0001)
    step = next(s for s in answer["next_steps"] if s["basis"] == "evidence")
    assert rules_0001["next_steps"][0]["text"] == "inspect process conditions"
    step["text"] = "Inspect the process conditions at the pump, first."
    fake_llm(wrap_reply(answer))
    d = api.diagnose_fault("obj_0001", use_llm=True)
    assert next(s for s in d["next_steps"] if s["priority"] == 1)["basis"] == "evidence"
    assert d["llm"]["filters"]["basis_downgraded"] == []


@pytest.mark.parametrize(
    "row,step,ok",
    [
        ("inspect process conditions", "Inspect the process conditions.", True),
        ("inspect process conditions", "INSPECT PROCESS-CONDITIONS (Current_State!R15)", True),
        ("inspect process conditions", "Inspect the processes and their conditions", True),
        ("inspect process conditions", "Inspect the suction strainer", False),
        ("inspect process conditions", "Check process conditions", False),  # verb matters
        ("the", "anything", False),  # a row with no meaningful words supports nothing
    ],
)
def test_supports_matches_meaningful_words(row, step, ok):
    assert llm_diagnose.supports(row, step) is ok


# --- model roles and profiles ------------------------------------------------ #
def test_live_role_uses_flash_next_profile():
    s = llm_client.load_settings()
    assert (s.role, s.model, s.timeout_s, s.max_tokens) == (
        "live",
        "Qwen/Qwen3.8-Flash-Next",
        60,
        4096,
    )
    assert (s.temperature, s.top_p, s.enable_thinking) == (0.7, 0.8, False)


def test_cache_role_uses_kimi_profile():
    s = llm_client.load_settings(llm_client.CACHE)
    assert (s.role, s.model, s.timeout_s, s.max_tokens) == (
        "cache",
        "moonshotai/Kimi-K3",
        300,
        32768,
    )
    assert (s.temperature, s.top_p, s.enable_thinking) == (1.0, 0.95, None)
    assert "extra_body" not in s.request_kwargs()  # Kimi always thinks: send no switch


def test_profiles_are_overridable_and_unknown_models_get_generic(monkeypatch):
    monkeypatch.setenv(llm_client.CACHE_MODEL_ENV, "deepseek-ai/DeepSeek-V4.1-Flash")
    monkeypatch.setenv(llm_client.TEMPERATURE_ENV, "0.5")
    monkeypatch.setenv(llm_client.TIMEOUT_ENV, "120")
    s = llm_client.load_settings(llm_client.CACHE)
    assert (s.model, s.temperature, s.top_p, s.timeout_s) == (
        "deepseek-ai/DeepSeek-V4.1-Flash",
        0.5,
        0.95,
        120,
    )
    monkeypatch.setenv(llm_client.MODEL_ENV, "some-org/unknown-model")
    monkeypatch.delenv(llm_client.TEMPERATURE_ENV)
    assert llm_client.profile_for("some-org/unknown-model") == llm_client.GENERIC_PROFILE
    s = llm_client.load_settings()
    assert (s.temperature, s.top_p, s.enable_thinking, s.max_tokens) == (0.7, 0.8, None, 4096)
    with pytest.raises(ValueError, match="unknown role"):
        llm_client.load_settings("batch")


def test_cli_builds_the_cache_with_the_cache_model(fake_llm, rules_0001, capsys, tmp_path):
    client = fake_llm(wrap_reply(fake_llm_answer(rules_0001)))
    assert llm_diagnose.main(["--refresh", "obj_0001"]) == 0
    (call,) = client.calls
    assert call["model"] == "moonshotai/Kimi-K3" and call["max_tokens"] == 32768
    out = capsys.readouterr().out
    assert out.startswith(
        "cache build: model=moonshotai/Kimi-K3 role=cache timeout=300 s max_tokens=32768\n"
    )
    assert "citation ok=True" in out
    written = json.loads(_cache_file(tmp_path).read_text(encoding="utf-8"))  # default dest
    assert (written["llm"]["model"], written["llm"]["role"]) == ("moonshotai/Kimi-K3", "cache")
    # Back outside the CLI, live calls use the live default again.
    assert llm_client.load_settings().model == "Qwen/Qwen3.8-Flash-Next"


def test_cli_never_writes_a_fallback(fake_llm, capsys, tmp_path):
    fake_llm("not json", "still not json")
    assert llm_diagnose.main(["obj_0001"]) == 1
    assert "REFUSED, nothing written" in capsys.readouterr().out
    assert not _cache_file(tmp_path).exists()


def test_cli_refuses_a_short_cache_timeout_before_calling(fake_llm, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv(llm_client.TIMEOUT_ENV, "60")
    client = fake_llm()
    assert llm_diagnose.main(["obj_0001"]) == 2
    assert "refused: cache timeout 60 s is below 300 s" in capsys.readouterr().out
    assert client.calls == [] and not _cache_file(tmp_path).exists()


def _llm_result(**rec):
    base = {"role": "cache", "model": "moonshotai/Kimi-K3", "timeout_s": 300}
    return {"method": "llm", "llm": {**base, **rec}}


@pytest.mark.parametrize(
    "result,problem",
    [
        (_llm_result(), None),
        (_llm_result(timeout_s=600), None),
        (_llm_result(role="live"), "made in role 'live'"),
        (_llm_result(model="Qwen/Qwen3-32B"), "made by 'Qwen/Qwen3-32B'"),
        (_llm_result(timeout_s=60), "timeout 60 s is below"),
        (
            {"method": "rule_based_fallback", "fallback_reason": "timeout after 300 s"},
            "not an LLM answer (rule_based_fallback: timeout after 300 s)",
        ),
    ],
)
def test_cache_build_problem(result, problem):
    settings = llm_client.load_settings(llm_client.CACHE)
    got = llm_diagnose.cache_build_problem(result, settings)
    assert got == problem if problem is None else problem in got


# --- the CLI exactly as run from a shell: python -m, against a local fake server --- #
class _FakeOpenAIServer:
    """Minimal OpenAI-compatible /v1/chat/completions on 127.0.0.1 (never the real API)."""

    def __init__(self, replies):
        import http.server

        self.replies, self.requests = list(replies), []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body})
                content = outer.replies.pop(0) if outer.replies else ""
                data = json.dumps(
                    {
                        "id": "fake",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _run_cli(server, out_dir, *extra_args, **env_overrides):
    """`python -m valmet.diagnostics.llm_diagnose ...` in a subprocess, pointed at `server`."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if not k.startswith(("VALMET_LLM", "FEATHERLESS"))}
    env.update(
        {name: "" for name in llm_client.SETTING_ENVS},  # set, so the repo .env can't win
        FEATHERLESS_API_KEY=FAKE_API_KEY,
        VALMET_LLM_BASE_URL=server.url,
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        **env_overrides,
    )
    env.pop(METADATA_DIR_ENV, None)
    cmd = [sys.executable, "-m", "valmet.diagnostics.llm_diagnose", "--refresh"]
    cmd += ["--out", str(out_dir), *extra_args, "obj_0001"]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)


@pytest.fixture
def fake_server():
    servers = []

    def start(*replies):
        servers.append(_FakeOpenAIServer(replies))
        return servers[-1]

    yield start
    for s in servers:
        s.close()


def test_python_dash_m_uses_the_cache_role(fake_server, rules_0001, tmp_path):
    server = fake_server(wrap_reply(fake_llm_answer(rules_0001)))
    proc = _run_cli(server, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.splitlines()[0] == (
        "cache build: model=moonshotai/Kimi-K3 role=cache timeout=300 s max_tokens=32768"
    )
    (req,) = server.requests
    assert req["path"] == "/v1/chat/completions"
    assert req["body"]["model"] == "moonshotai/Kimi-K3" and req["body"]["max_tokens"] == 32768
    assert "chat_template_kwargs" not in req["body"]  # Kimi profile: thinking switch omitted
    written = json.loads((tmp_path / "obj_0001.json").read_text(encoding="utf-8"))
    rec = written["llm"]
    assert (rec["model"], rec["role"], rec["timeout_s"]) == ("moonshotai/Kimi-K3", "cache", 300)
    assert written["method"] == "llm" and written["citation_check"]["ok"]
    assert FAKE_API_KEY not in proc.stdout + proc.stderr


def test_python_dash_m_live_model_env_does_not_leak_into_cache_build(
    fake_server, rules_0001, tmp_path
):
    """The bug that built the cache with Qwen3-32B: a live-model setting must not be used."""
    server = fake_server(wrap_reply(fake_llm_answer(rules_0001)))
    proc = _run_cli(server, tmp_path, VALMET_LLM_MODEL="Qwen/Qwen3-32B")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert server.requests[0]["body"]["model"] == "moonshotai/Kimi-K3"


def test_python_dash_m_writes_nothing_on_fallback(fake_server, tmp_path):
    server = fake_server("not json", "still not json")
    proc = _run_cli(server, tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "REFUSED, nothing written" in proc.stdout
    assert len(server.requests) == 2 and not (tmp_path / "obj_0001.json").exists()


def test_python_dash_m_refuses_short_timeout_without_calling(fake_server, tmp_path):
    server = fake_server()
    proc = _run_cli(server, tmp_path, VALMET_LLM_TIMEOUT_S="60")
    assert proc.returncode == 2
    assert "timeout=60 s" in proc.stdout and "refused: cache timeout 60 s" in proc.stdout
    assert server.requests == [] and not any(tmp_path.iterdir())


def test_cached_answer_is_reused_whatever_model_made_it(fake_llm, rules_0001, monkeypatch):
    client = fake_llm(wrap_reply(fake_llm_answer(rules_0001)))
    with llm_diagnose.use_role(llm_client.CACHE):
        built = api.diagnose_fault("obj_0001", use_llm=True, refresh=True)
    assert built["llm"]["model"] == "moonshotai/Kimi-K3"
    # Live default (Flash-Next) and an arbitrary live model both get the Kimi answer.
    assert api.diagnose_fault("obj_0001", use_llm=True) == built
    monkeypatch.setenv(llm_client.MODEL_ENV, "some-org/other-model")
    assert api.diagnose_fault("obj_0001", use_llm=True) == built
    assert len(client.calls) == 1


def test_background_sentence_with_a_number_is_dropped(fake_llm, rules_0001):
    answer = fake_llm_answer(rules_0001)
    answer["explanation"] = (
        "Vibration is high (Telemetry_Sample_3d!R2017). "
        "General background: NPSH margin should exceed 1.5 m. "
        "General background: collapsing vapour bubbles erode the impeller."
    )
    fake_llm(wrap_reply(answer))
    d = api.diagnose_fault("obj_0001", use_llm=True)
    assert d["explanation"] == (
        "Vibration is high (Telemetry_Sample_3d!R2017). "
        "General background: collapsing vapour bubbles erode the impeller."
    )
    assert d["llm"]["filters"]["background_sentences_dropped"] == [
        {"where": "explanation", "sentence": "General background: NPSH margin should exceed 1.5 m."}
    ]


def test_clean_answer_passes_the_filters_unchanged(fake_llm, rules_0001):
    fake_llm(wrap_reply(fake_llm_answer(rules_0001)))
    filters = api.diagnose_fault("obj_0001", use_llm=True)["llm"]["filters"]
    assert filters == {
        "background_sentences_dropped": [],
        "safety_dropped": [],
        "safety_added": [],
        "basis_downgraded": [],
    }


def test_invalid_json_twice_falls_back(fake_llm, rules_0001, tmp_path):
    client = fake_llm("not json at all", '{"summary": "still broken",')
    d = api.diagnose_fault("obj_0001", use_llm=True)
    assert d["method"] == "rule_based_fallback"
    assert d["fallback_reason"].startswith("invalid reply after retry")
    assert len(client.calls) == 2
    retry = client.calls[1]["messages"]
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert "no JSON object" in retry[-1]["content"]
    assert d["summary"] == rules_0001["summary"]
    assert not _cache_file(tmp_path).exists()  # fallbacks are never cached


def test_schema_error_then_valid_reply_succeeds(fake_llm, rules_0001):
    answer = fake_llm_answer(rules_0001)
    broken = {k: v for k, v in answer.items() if k != "safety_precautions"}
    client = fake_llm(json.dumps(broken), wrap_reply(answer))
    d = api.diagnose_fault("obj_0001", use_llm=True)
    assert d["method"] == "llm" and d["llm"]["attempts"] == 2
    assert "safety_precautions" in client.calls[1]["messages"][-1]["content"]
    assert d["llm"]["retry_reason"].startswith("JSON does not match the schema")
    assert "safety_precautions" in d["llm"]["retry_reason"]


def test_empty_reply_falls_back(fake_llm):
    client = fake_llm("   ")
    d = api.diagnose_fault("obj_0001", use_llm=True)
    assert d["method"] == "rule_based_fallback"
    assert d["fallback_reason"].startswith("empty reply")
    assert len(client.calls) == 1


def _timeouts():
    out = [TimeoutError("read timed out")]
    try:
        import httpx
        import openai

        out.append(openai.APITimeoutError(request=httpx.Request("POST", "https://x.invalid")))
    except ImportError:
        pass
    return out


@pytest.mark.parametrize("error", _timeouts(), ids=lambda e: type(e).__name__)
def test_timeout_falls_back(fake_llm, error):
    fake_llm(error)
    d = api.diagnose_fault("obj_0001", use_llm=True)
    assert d["method"] == "rule_based_fallback"
    assert d["fallback_reason"] == "timeout after 60 s"


class _SlowClient:
    """create() blocks far longer than the deadline, like the 252 s Featherless call."""

    def __init__(self):
        self.release = threading.Event()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        self.release.wait(30)
        raise AssertionError("should have been abandoned")


def test_hard_wall_clock_deadline(fake_llm, monkeypatch):
    monkeypatch.setenv(llm_client.TIMEOUT_ENV, "0.3")
    slow = _SlowClient()
    monkeypatch.setattr(llm_client, "make_client", lambda _s: slow)
    start = time.perf_counter()
    d = api.diagnose_fault("obj_0001", use_llm=True)
    elapsed = time.perf_counter() - start
    slow.release.set()  # let the abandoned daemon thread finish
    assert d["method"] == "rule_based_fallback"
    assert d["fallback_reason"] == "timeout after 0.3 s"
    assert elapsed < 5


@pytest.mark.parametrize("value", ["0", "-5", "abc", "4000"])
def test_bad_timeout_setting_falls_back(fake_llm, monkeypatch, value):
    monkeypatch.setenv(llm_client.TIMEOUT_ENV, value)
    client = fake_llm()
    d = api.diagnose_fault("obj_0001", use_llm=True)
    assert d["method"] == "rule_based_fallback"
    assert "VALMET_LLM_TIMEOUT_S" in d["fallback_reason"] and client.calls == []


def test_max_tokens_setting_is_used_and_recorded(fake_llm, rules_0001, monkeypatch):
    client = fake_llm(wrap_reply(fake_llm_answer(rules_0001)))
    assert api.diagnose_fault("obj_0001", use_llm=True)["llm"]["max_tokens"] == 4096
    monkeypatch.setenv(llm_client.MAX_TOKENS_ENV, "32768")
    client.replies.append(wrap_reply(fake_llm_answer(rules_0001)))
    d = api.diagnose_fault("obj_0001", use_llm=True, refresh=True)
    assert client.calls[-1]["max_tokens"] == 32768 and d["llm"]["max_tokens"] == 32768


@pytest.mark.parametrize("value", ["0", "1.5", "lots", "999999"])
def test_bad_max_tokens_setting_falls_back(fake_llm, monkeypatch, value):
    monkeypatch.setenv(llm_client.MAX_TOKENS_ENV, value)
    client = fake_llm()
    d = api.diagnose_fault("obj_0001", use_llm=True)
    assert "VALMET_LLM_MAX_TOKENS" in d["fallback_reason"] and client.calls == []


def test_timeout_setting_is_used_and_recorded(fake_llm, rules_0001, monkeypatch):
    monkeypatch.setenv(llm_client.TIMEOUT_ENV, "300")
    fake_llm(wrap_reply(fake_llm_answer(rules_0001)))
    assert api.diagnose_fault("obj_0001", use_llm=True)["llm"]["timeout_s"] == 300


def test_api_error_falls_back_without_leaking_details(fake_llm):
    err = RuntimeError(f"401 bad key {FAKE_API_KEY}")
    err.status_code = 401
    fake_llm(err)
    d = api.diagnose_fault("obj_0001", use_llm=True)
    assert d["fallback_reason"] == "API error: RuntimeError (HTTP 401)"
    assert FAKE_API_KEY not in json.dumps(d)


def test_second_call_uses_cache_and_refresh_bypasses_it(fake_llm, rules_0001, tmp_path):
    client = fake_llm(wrap_reply(fake_llm_answer(rules_0001)))
    first = api.diagnose_fault("obj_0001", use_llm=True)
    second = api.diagnose_fault("obj_0001", use_llm=True)
    assert len(client.calls) == 1
    assert second == first and _cache_file(tmp_path).exists()

    client.replies.append(wrap_reply(fake_llm_answer(rules_0001)))
    third = api.diagnose_fault("obj_0001", use_llm=True, refresh=True)
    assert len(client.calls) == 2 and third["method"] == "llm"


def test_stale_cache_is_regenerated(fake_llm, rules_0001, tmp_path):
    client = fake_llm(wrap_reply(fake_llm_answer(rules_0001)))
    api.diagnose_fault("obj_0001", use_llm=True)
    cached = json.loads(_cache_file(tmp_path).read_text(encoding="utf-8"))
    cached["llm"]["input_sha256"] = "0" * 64  # as if the Excel or the rules had changed
    _cache_file(tmp_path).write_text(json.dumps(cached), encoding="utf-8")

    client.replies.append(wrap_reply(fake_llm_answer(rules_0001)))
    api.diagnose_fault("obj_0001", use_llm=True)
    assert len(client.calls) == 2


def test_offline_uses_cache_else_rules(fake_llm, rules_0001):
    offline = api.diagnose_fault("obj_0001", use_llm=True, offline=True)
    assert offline["method"] == "rule_based_fallback"
    assert offline["fallback_reason"] == "offline and no cached LLM diagnosis"

    client = fake_llm(wrap_reply(fake_llm_answer(rules_0001)))
    live = api.diagnose_fault("obj_0001", use_llm=True)
    assert api.diagnose_fault("obj_0001", use_llm=True, offline=True) == live
    assert len(client.calls) == 1  # offline never calls the API


def test_settings_overridable_by_env(fake_llm, rules_0001, monkeypatch):
    monkeypatch.setenv(llm_client.MODEL_ENV, "some-org/other-model")
    monkeypatch.setenv(llm_client.TEMPERATURE_ENV, "0.2")
    monkeypatch.setenv(llm_client.TOP_P_ENV, "0.95")
    monkeypatch.setenv(llm_client.THINKING_ENV, "omit")
    client = fake_llm(wrap_reply(fake_llm_answer(rules_0001)))
    d = api.diagnose_fault("obj_0001", use_llm=True)
    (call,) = client.calls
    assert call["model"] == "some-org/other-model"
    assert (call["temperature"], call["top_p"]) == (0.2, 0.95)
    assert "extra_body" not in call
    assert d["llm"]["enable_thinking"] is None


def test_bad_setting_falls_back(fake_llm, monkeypatch):
    monkeypatch.setenv(llm_client.TEMPERATURE_ENV, "warm")
    client = fake_llm()
    d = api.diagnose_fault("obj_0001", use_llm=True)
    assert d["method"] == "rule_based_fallback"
    assert "VALMET_LLM_TEMPERATURE" in d["fallback_reason"]
    assert client.calls == []


@pytest.mark.parametrize(
    "text",
    [
        '<think>x</think>{"a": 1}',
        '<THINK>\n{"a": 2}\n</THINK>\n```json\n{"a": 1}\n```',
        'Sure! {"a": 1} and then {"b": 2}',
    ],
)
def test_parse_reply_finds_the_first_json_object(text):
    with pytest.raises(llm_diagnose.ReplyError, match="schema") as err:
        llm_diagnose.parse_reply(text)  # {"a": 1} is valid JSON but not a diagnosis
    assert "summary" in str(err.value)


def test_parse_reply_unclosed_think_is_an_error():
    with pytest.raises(llm_diagnose.ReplyError, match="no JSON object"):
        llm_diagnose.parse_reply('<think>{"summary": "never finished thinking"')


@pytest.mark.parametrize("object_id", ALL_OBJECTS)
def test_rule_based_results_pass_the_citation_checker(object_id):
    report = check_citations(api.diagnose_fault(object_id))
    assert report["ok"], report["failed"]
    assert report["checked"] > 0


def test_citation_checker_flags_unknown_source_and_field():
    d = api.diagnose_fault("obj_0001")
    d["evidence"] += [
        {"source": "Current_State!R9999", "field": "health_score", "value": 45},
        {"source": "Current_State!R15", "field": "no_such_column", "value": 1},
        {"source": "Current_State!R2", "field": "health_score", "value": 0},  # other device
    ]
    kinds = [f["kind"] for f in check_citations(d)["failed"]]
    assert kinds == ["missing_source", "unknown_field", "not_in_object_evidence"]


# --------------------------------------------------------------------------- #
# get_asset / get_nearby_assets
# --------------------------------------------------------------------------- #
def test_get_asset_has_sources():
    a = api.get_asset("obj_0001")
    assert a["object_id"] == "obj_0001" and a["device_id"] == "ASSET-014"
    assert a["device"]["_source"] == "Asset_Master!R15"
    assert a["condition"]["_source"] == "Current_State!R15"
    assert a["condition"]["issue_code"] == "CAVITATION_RISK"


def test_nearby_keeps_order_and_skips_unknown():
    ids = ["obj_0003", "obj_9999", "obj_0001", None, "obj_014", "obj_0002"]
    out = api.get_nearby_assets(ids)
    assert [a["object_id"] for a in out] == ["obj_0003", "obj_0001", "obj_0002"]
    assert (
        out[1]["status_line"]
        == "P-002 process pump — attention required, health 45 (Current_State!R15)"
    )
    for a in out:
        assert a["status_line"].startswith(a["device"]["tag"])
        assert a["condition"]["_source"] in a["status_line"]


def test_nearby_empty():
    assert api.get_nearby_assets([]) == []


def test_unknown_object_raises_for_single_lookups():
    with pytest.raises(KeyError, match="obj_9999"):
        api.get_asset("obj_9999")
    with pytest.raises(KeyError, match="obj_9999"):
        api.diagnose_fault("obj_9999")


@pytest.mark.parametrize("object_id", ALL_OBJECTS)
def test_all_outputs_json_serialisable(object_id):
    json.dumps(api.get_asset(object_id))
    json.dumps(api.diagnose_fault(object_id))
    json.dumps(api.get_nearby_assets([object_id, "obj_9999"]))


def test_api_exposes_exactly_three_functions():
    public = {n for n in dir(api) if not n.startswith("_") and callable(getattr(api, n))}
    assert public == {"get_asset", "get_nearby_assets", "diagnose_fault"}
    assert set(api.__all__) == public
