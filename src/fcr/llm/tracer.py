"""Structured execution traces.

Ported from Project 2 (src/tracing/tracer.py) in C2. Kept: append-only JSONL,
flush per event, payload truncation, node spans, LLM-call events, a reader
that tolerates a half-written last line. Removed: Project 2's agent-specific
events (controller routes, retrieval calls, the D1-D5 decision-surface enum).
Generalised: `decision()` takes any name, because this project's decisions are
different ones (the verifier's verdict, regenerate or fall back to the
template), and C20's check is that a trace shows a regeneration that an
injected failure triggered.

Why traces matter here
----------------------
1. Audit. A released letter must be traceable to the attributions, the
   generation call, and the verifier verdicts that let it through. Stored
   traces are that record.
2. Evaluation. C25's first-attempt versus final pass rates for E2 and E3 are
   counted from regeneration decisions in traces, not reconstructed later.
3. Debugging. A long resumable run that dies keeps every event up to the
   point of death, because each one is flushed on write.

Format: one JSONL file per run, named by trace_id, one JSON object per line.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

TRACE_FORMAT_VERSION = 1

# Payload guards: generous enough to keep a letter and its claims legible,
# tight enough that hundreds of traces stay browsable.
MAX_STRING_CHARS = 2000
MAX_LIST_ITEMS = 25


class EventType(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    NODE_ENTER = "node_enter"
    NODE_EXIT = "node_exit"
    DECISION = "decision"
    LLM_CALL = "llm_call"
    ERROR = "error"
    NOTE = "note"


def _truncate(value: Any, _depth: int = 0) -> Any:
    """Bound payload size while keeping structure recognisable."""
    if _depth > 6:
        return "<max depth>"
    if isinstance(value, str):
        if len(value) > MAX_STRING_CHARS:
            return value[:MAX_STRING_CHARS] + f"... <+{len(value) - MAX_STRING_CHARS} chars>"
        return value
    if isinstance(value, dict):
        return {str(k): _truncate(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_truncate(v, _depth + 1) for v in list(value)[:MAX_LIST_ITEMS]]
        if len(value) > MAX_LIST_ITEMS:
            items.append(f"<+{len(value) - MAX_LIST_ITEMS} more>")
        return items
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, Enum):
        return value.value
    return _truncate(str(value), _depth)


@dataclass
class TraceEvent:
    seq: int
    ts: str
    elapsed_ms: float
    type: str
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    def to_json(self) -> str:
        d: dict[str, Any] = {
            "seq": self.seq,
            "ts": self.ts,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "type": self.type,
            "name": self.name,
        }
        if self.duration_ms is not None:
            d["duration_ms"] = round(self.duration_ms, 2)
        if self.payload:
            d["payload"] = self.payload
        return json.dumps(d, ensure_ascii=False, default=str)


class Tracer:
    """Writes one JSONL trace file for one run."""

    def __init__(
        self,
        trace_dir: str | Path,
        *,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        enabled: bool = True,
        now: Any = time.time,
    ) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.enabled = enabled
        self._now = now
        self._t0 = now()
        self._seq = 0
        self._lock = threading.RLock()
        self._closed = False
        self._node_stack: list[str] = []

        self.dir = Path(trace_dir)
        self.path = self.dir / f"{self.trace_id}.jsonl"
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")  # truncate any prior file for this id
            self.emit(
                EventType.RUN_START,
                "run",
                trace_format_version=TRACE_FORMAT_VERSION,
                trace_id=self.trace_id,
                **(metadata or {}),
            )

    # -- core ---------------------------------------------------------------

    def emit(
        self,
        event_type: EventType,
        name: str,
        *,
        duration_ms: float | None = None,
        **payload: Any,
    ) -> TraceEvent | None:
        if not self.enabled or self._closed:
            return None
        with self._lock:
            self._seq += 1
            if self._node_stack:
                payload.setdefault("node", self._node_stack[-1])
            event = TraceEvent(
                seq=self._seq,
                ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                elapsed_ms=(self._now() - self._t0) * 1000.0,
                type=event_type.value,
                name=name,
                payload=_truncate(payload),
                duration_ms=duration_ms,
            )
            # Flush per event: a run killed mid-eval must keep what it did.
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(event.to_json() + "\n")
                fh.flush()
            return event

    # -- spans --------------------------------------------------------------

    @contextmanager
    def node(self, name: str, **payload: Any) -> Iterator[Tracer]:
        """Wrap a pipeline step: enter, exit with duration, or error."""
        self.emit(EventType.NODE_ENTER, name, **payload)
        self._node_stack.append(name)
        start = self._now()
        try:
            yield self
        except Exception as exc:
            self._node_stack.pop()
            self.emit(
                EventType.ERROR,
                name,
                duration_ms=(self._now() - start) * 1000.0,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        else:
            self._node_stack.pop()
            self.emit(EventType.NODE_EXIT, name, duration_ms=(self._now() - start) * 1000.0)

    # -- typed events -------------------------------------------------------

    def decision(
        self,
        name: str,
        *,
        chosen: Any,
        alternatives: list[Any] | None = None,
        rationale: str = "",
        inputs: dict[str, Any] | None = None,
    ) -> None:
        """Record a choice the pipeline made, with what else it could have done.

        Examples in this project: name="verifier" chosen="fail" with the
        failure modes as inputs; name="regenerate" chosen="retry" or
        "template_fallback".
        """
        self.emit(
            EventType.DECISION,
            name,
            chosen=chosen,
            alternatives=alternatives or [],
            rationale=rationale,
            inputs=inputs or {},
        )

    def llm_call(
        self,
        *,
        role: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_s: float,
        from_cache: bool,
        fell_back: bool = False,
        attempts: int = 1,
    ) -> None:
        self.emit(
            EventType.LLM_CALL,
            model,
            duration_ms=latency_s * 1000.0,
            role=role,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            from_cache=from_cache,
            fell_back=fell_back,
            attempts=attempts,
        )

    def note(self, name: str, **payload: Any) -> None:
        self.emit(EventType.NOTE, name, **payload)

    def close(self, **payload: Any) -> None:
        if not self.enabled or self._closed:
            return
        self.emit(EventType.RUN_END, "run", duration_ms=(self._now() - self._t0) * 1000.0, **payload)
        self._closed = True

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.emit(EventType.ERROR, "run", error_type=exc_type.__name__, error=str(exc))
        self.close(ok=exc is None)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


class TraceReader:
    """Loads a trace and answers the questions the evaluation needs to ask."""

    def __init__(self, events: list[dict[str, Any]], path: Path | None = None) -> None:
        self.events = events
        self.path = path

    @classmethod
    def load(cls, path: str | Path) -> TraceReader:
        path = Path(path)
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # A process killed mid-write leaves one partial line. Losing
                # that event beats losing the whole trace.
                continue
        return cls(events, path)

    # -- queries ------------------------------------------------------------

    def of_type(self, event_type: EventType) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == event_type.value]

    @property
    def trace_id(self) -> str | None:
        starts = self.of_type(EventType.RUN_START)
        return starts[0].get("payload", {}).get("trace_id") if starts else None

    def nodes_visited(self) -> list[str]:
        return [e["name"] for e in self.of_type(EventType.NODE_ENTER)]

    def decisions(self, name: str | None = None) -> list[dict[str, Any]]:
        out = self.of_type(EventType.DECISION)
        if name is not None:
            out = [e for e in out if e.get("name") == name]
        return out

    def decision_counts(self) -> dict[str, dict[str, int]]:
        """{decision name: {chosen value: count}} — e.g. how often regeneration fired."""
        counts: dict[str, Counter] = {}
        for e in self.of_type(EventType.DECISION):
            chosen = e.get("payload", {}).get("chosen")
            counts.setdefault(e["name"], Counter())[str(chosen)] += 1
        return {name: dict(c) for name, c in counts.items()}

    def llm_totals(self) -> dict[str, Any]:
        calls = self.of_type(EventType.LLM_CALL)
        live = [c for c in calls if not c.get("payload", {}).get("from_cache")]
        return {
            "calls": len(calls),
            "live_calls": len(live),
            "cached_calls": len(calls) - len(live),
            "total_tokens": sum(c.get("payload", {}).get("total_tokens", 0) for c in calls),
            "live_tokens": sum(c.get("payload", {}).get("total_tokens", 0) for c in live),
            "by_model": {m: sum(1 for c in calls if c["name"] == m) for m in {c["name"] for c in calls}},
        }

    def errors(self) -> list[dict[str, Any]]:
        return self.of_type(EventType.ERROR)

    def wall_clock_ms(self) -> float:
        ends = self.of_type(EventType.RUN_END)
        if ends:
            return ends[-1].get("duration_ms", 0.0)
        return self.events[-1]["elapsed_ms"] if self.events else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "events": len(self.events),
            "nodes": self.nodes_visited(),
            "decisions": self.decision_counts(),
            "llm": self.llm_totals(),
            "errors": len(self.errors()),
            "wall_clock_ms": round(self.wall_clock_ms(), 1),
        }
