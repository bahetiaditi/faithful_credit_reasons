"""Tests for structured tracing (ported from Project 2 in C2).

Kept: format, spans, truncation, robustness. Replaced: Project 2's routing
and decision-surface tests, with generic decision tests for this project's
verify-and-regenerate loop.
"""
import json

import pytest

from fcr.llm.tracer import (
    MAX_LIST_ITEMS,
    MAX_STRING_CHARS,
    EventType,
    TraceReader,
    Tracer,
)


@pytest.fixture
def tracer(tmp_path):
    return Tracer(tmp_path / "traces", trace_id="t1", metadata={"applicant": "case-7"})


def read(tracer):
    return TraceReader.load(tracer.path)


# --- format ----------------------------------------------------------------


def test_every_line_is_valid_json(tracer):
    with tracer.node("planner"):
        tracer.note("hello", x=1)
    tracer.close()
    for line in tracer.path.read_text().splitlines():
        json.loads(line)


def test_sequence_is_monotonic(tracer):
    for i in range(5):
        tracer.note(f"n{i}")
    tracer.close()
    seqs = [e["seq"] for e in read(tracer).events]
    assert seqs == sorted(seqs) == list(range(1, len(seqs) + 1))


def test_file_named_by_trace_id(tracer):
    assert tracer.path.name == "t1.jsonl"


def test_disabled_tracer_writes_nothing(tmp_path):
    t = Tracer(tmp_path / "tr", trace_id="x", enabled=False)
    t.note("ignored")
    t.close()
    assert not (tmp_path / "tr" / "x.jsonl").exists()


# --- node spans ------------------------------------------------------------


def test_node_span_emits_enter_and_exit_with_duration(tracer):
    with tracer.node("retriever"):
        pass
    tracer.close()
    r = read(tracer)
    assert r.nodes_visited() == ["retriever"]
    assert r.of_type(EventType.NODE_EXIT)[0]["duration_ms"] >= 0


def test_events_inside_a_span_are_attributed_to_that_node(tracer):
    with tracer.node("critic"):
        tracer.note("checking")
    tracer.close()
    note = read(tracer).of_type(EventType.NOTE)[0]
    assert note["payload"]["node"] == "critic"


def test_exception_in_node_is_recorded_and_reraised(tracer):
    with pytest.raises(ValueError):
        with tracer.node("planner"):
            raise ValueError("bad plan")
    tracer.close()
    errors = read(tracer).errors()
    assert errors[0]["payload"]["error_type"] == "ValueError"
    assert "bad plan" in errors[0]["payload"]["error"]


def test_node_stack_unwinds_after_error(tracer):
    with pytest.raises(ValueError):
        with tracer.node("a"):
            raise ValueError("x")
    tracer.note("after")
    tracer.close()
    # 'after' must not still be attributed to the failed node. With the stack
    # unwound there is no node key at all, so the payload is omitted entirely.
    note = read(tracer).of_type(EventType.NOTE)[0]
    assert note.get("payload", {}).get("node") != "a"


# --- decision surfaces (the C35 ablation check) ----------------------------


# --- llm and retrieval -----------------------------------------------------


def test_llm_totals_separate_live_from_cached(tracer):
    tracer.llm_call(role="generation", model="m", provider="groq",
                    prompt_tokens=100, completion_tokens=50, total_tokens=150,
                    latency_s=1.0, from_cache=False)
    tracer.llm_call(role="generation", model="m", provider="groq",
                    prompt_tokens=100, completion_tokens=50, total_tokens=150,
                    latency_s=0.0, from_cache=True)
    tracer.close()
    totals = read(tracer).llm_totals()
    assert totals == {
        "calls": 2, "live_calls": 1, "cached_calls": 1,
        "total_tokens": 300, "live_tokens": 150, "by_model": {"m": 2},
    }


# --- payload guards --------------------------------------------------------


def test_long_strings_are_truncated(tracer):
    tracer.note("big", blob="x" * (MAX_STRING_CHARS + 500))
    tracer.close()
    blob = read(tracer).of_type(EventType.NOTE)[0]["payload"]["blob"]
    assert len(blob) < MAX_STRING_CHARS + 200
    assert "+500 chars" in blob


def test_long_lists_are_truncated(tracer):
    tracer.note("big", items=list(range(MAX_LIST_ITEMS + 10)))
    tracer.close()
    items = read(tracer).of_type(EventType.NOTE)[0]["payload"]["items"]
    assert len(items) == MAX_LIST_ITEMS + 1
    assert "more" in str(items[-1])


def test_nested_payloads_are_truncated(tracer):
    tracer.note("nested", d={"inner": {"blob": "y" * (MAX_STRING_CHARS + 100)}})
    tracer.close()
    blob = read(tracer).of_type(EventType.NOTE)[0]["payload"]["d"]["inner"]["blob"]
    assert "chars" in blob


# --- robustness ------------------------------------------------------------


def test_truncated_final_line_does_not_lose_the_trace(tracer):
    """A process killed mid-write leaves a partial line. Losing that one
    event must not cost the whole file."""
    tracer.note("good")
    tracer.close()
    with tracer.path.open("a") as fh:
        fh.write('{"seq": 99, "type": "note", "na')
    r = read(tracer)
    assert len(r.events) >= 2
    assert any(e["name"] == "good" for e in r.of_type(EventType.NOTE))


def test_context_manager_records_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with Tracer(tmp_path / "tr", trace_id="cm") as t:
            raise RuntimeError("exploded")
    r = TraceReader.load(tmp_path / "tr" / "cm.jsonl")
    assert r.errors()
    assert r.of_type(EventType.RUN_END)[0]["payload"]["ok"] is False


def test_run_start_records_trace_id_and_metadata(tracer):
    tracer.close()
    start = read(tracer).of_type(EventType.RUN_START)[0]
    assert start["payload"]["applicant"] == "case-7"
    assert start["payload"]["trace_id"] == "t1"


# --- decisions (this project's verify-and-regenerate loop) -----------------


def test_decision_records_alternatives_rationale_and_inputs(tracer):
    tracer.decision(
        "verifier", chosen="fail", alternatives=["pass", "fail"],
        rationale="1 sign error", inputs={"failure_modes": ["sign_error"]},
    )
    tracer.close()
    d = read(tracer).decisions("verifier")[0]["payload"]
    assert d["chosen"] == "fail"
    assert d["alternatives"] == ["pass", "fail"]
    assert d["inputs"]["failure_modes"] == ["sign_error"]


def test_decision_counts_show_how_often_regeneration_fired(tracer):
    """C20/C25 count first-attempt vs final passes from these events."""
    tracer.decision("verifier", chosen="fail")
    tracer.decision("regenerate", chosen="retry")
    tracer.decision("verifier", chosen="fail")
    tracer.decision("regenerate", chosen="template_fallback")
    tracer.close()
    counts = read(tracer).decision_counts()
    assert counts["verifier"] == {"fail": 2}
    assert counts["regenerate"] == {"retry": 1, "template_fallback": 1}
    assert read(tracer).decisions() and not read(tracer).decisions("missing")


def test_summary_shape(tracer):
    with tracer.node("generate"):
        tracer.decision("verifier", chosen="pass")
    tracer.close()
    s = read(tracer).summary()
    assert s["trace_id"] == "t1"
    assert s["nodes"] == ["generate"]
    assert s["decisions"] == {"verifier": {"pass": 1}}
