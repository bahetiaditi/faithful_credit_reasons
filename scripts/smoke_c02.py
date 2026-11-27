"""C2 smoke run: the ported LLM layer against the real APIs.

    python scripts/smoke_c02.py             # live; needs GOOGLE_API_KEY and GROQ_API_KEY in .env
    python scripts/smoke_c02.py --dry-run   # same checks on scripted fake backends, no network

Checks, each printed as PASS or FAIL:
  1. every configured model answers (catches a renamed or retired model id)
  2. an identical second call is served from the cache and spends no quota
  3. when the primary generation model fails, the call falls through to the next
  4. a structured call returns parseable JSON with truncation repair switched off
  5. the trace accounts for every call, live and cached

Live calls go through the real quota ledger (.cache/quota_ledger.json) so the
spend is counted. Each check uses a fresh temporary cache, so every live check
really calls the API. Expected spend: about 8 short calls.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from fcr.config import has_env, load_config, resolve_path
from fcr.llm.cache import PromptCache
from fcr.llm.provider import LLMProvider, ProviderError, Role
from fcr.llm.rate_limit import ModelLimits, QuotaLedger
from fcr.llm.tracer import TraceReader, Tracer

PROBE_SYSTEM = "You are a terse assistant."
PROBE_USER = "Reply with exactly the word OK and nothing else."
CLAIMS_SYSTEM = "You extract claims from adverse-action letters. Reply with JSON only."
CLAIMS_USER = (
    'Letter: "Your application was declined mainly because of recent missed payments '
    'and a high level of outstanding debt."\n'
    'Return {"claims": [{"factor": <short name>, "direction": "increases_risk"}]} '
    "with one entry per factor the letter states."
)


class _Unavailable(Exception):
    """Stands in for a provider 503 in the fallback check."""

    status_code = 503


class FakeBackend:
    """Dry-run stand-in: answers OK, or valid claim JSON when JSON is requested."""

    def __init__(self, name: str) -> None:
        self.name = name

    def complete(self, spec, system, user, temperature, max_output_tokens, json_mode):
        if "claims" in user:
            text = '{"claims": [{"factor": "missed payments", "direction": "increases_risk"}]}'
        else:
            text = "OK"
        return text, 40, 5


class FailFor:
    """Wraps a backend and fails every call to one model, to force a fallback."""

    def __init__(self, inner, failing_model: str) -> None:
        self.inner = inner
        self.name = inner.name
        self.failing_model = failing_model

    def complete(self, spec, *args, **kwargs):
        if spec.model == self.failing_model:
            raise _Unavailable(f"simulated 503 for {spec.model}")
        return self.inner.complete(spec, *args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="use fake backends; no network, no keys")
    args = parser.parse_args()

    cfg = load_config()
    llm = cfg.llm
    results: list[tuple[str, bool, str]] = []
    work = Path(tempfile.mkdtemp(prefix="fcr_smoke_c02_"))

    def record(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    # One ledger shared by every provider below. Separate instances would each
    # hold their own in-memory copy and overwrite one another's records.
    limits = {m.model: ModelLimits.from_config(m.limits.model_dump()) for ch in llm.roles.values() for m in ch}
    if args.dry_run:
        print("mode: DRY RUN (fake backends, temporary ledger)")
        ledger = QuotaLedger(work / "ledger.json", limits)
        backends = {"groq": FakeBackend("groq"), "gemini": FakeBackend("gemini")}
    else:
        missing = [k for k in ("GOOGLE_API_KEY", "GROQ_API_KEY") if not has_env(k)]
        if missing:
            print(f"missing keys {missing}: copy .env.example to .env at the repo root and fill them in")
            return 1
        print("mode: LIVE")
        ledger = QuotaLedger(resolve_path(llm.quota_ledger_path), limits)
        backends = None  # the provider creates real backends lazily

    def make(*, pin: dict[str, str] | None = None, cache_dir: Path | None = None, wrap=None, tracer=None):
        provider = LLMProvider(
            llm.model_copy(update={"pin": {**llm.pin, **(pin or {})}}),
            backends=dict(backends) if backends is not None else None,
            ledger=ledger,
            cache=PromptCache(cache_dir or work / "cache_unused", enabled=cache_dir is not None),
            sleep=(lambda s: None) if args.dry_run else __import__("time").sleep,
            tracer=tracer,
        )
        if wrap is not None:
            for name in ("groq", "gemini"):
                provider._backends[name] = wrap(provider._backend(name))
        return provider

    # 1. every model answers
    print("\n== 1. Every configured model answers")
    for role in Role:
        for spec in llm.roles.get(role.value, ()):
            try:
                resp = make(pin={role.value: spec.model}).complete(PROBE_SYSTEM, PROBE_USER, role=role, use_cache=False)
                ok = bool(resp.text.strip()) and resp.model == spec.model
                record(f"{role.value}/{spec.model}", ok, f"{resp.text.strip()[:30]!r}, {resp.total_tokens} tokens")
            except Exception as exc:  # report and keep checking the other models
                record(f"{role.value}/{spec.model}", False, f"{type(exc).__name__}: {str(exc)[:200]}")

    tracer = Tracer(work / "traces", trace_id="smoke_c02")

    # 2. cache: the second identical call is free
    print("\n== 2. Identical call is served from the cache")
    provider = make(cache_dir=work / "cache_check", tracer=tracer)
    first = provider.complete(PROBE_SYSTEM, PROBE_USER + " (cache check)", role=Role.EXTRACTION)
    used_after_first = provider.ledger.snapshot()[first.model]["rpd"][0]
    second = provider.complete(PROBE_SYSTEM, PROBE_USER + " (cache check)", role=Role.EXTRACTION)
    used_after_second = provider.ledger.snapshot()[first.model]["rpd"][0]
    record(
        "cache",
        (not first.from_cache) and second.from_cache and used_after_second == used_after_first,
        f"first live={not first.from_cache}, second cached={second.from_cache}, "
        f"requests recorded {used_after_first} -> {used_after_second}",
    )

    # 3. fallback: the primary generation model fails, the next one answers
    print("\n== 3. Fallback when the primary generation model fails")
    chain = [s.model for s in llm.roles["generation"]]
    try:
        provider = make(wrap=lambda b: FailFor(b, chain[0]), tracer=tracer)
        resp = provider.complete(PROBE_SYSTEM, PROBE_USER, role=Role.GENERATION, use_cache=False)
        record("fallback", resp.fell_back and resp.model == chain[1],
               f"{chain[0]} failed -> answered by {resp.model} after fell_back={resp.fell_back}")
    except (ProviderError, Exception) as exc:
        record("fallback", False, f"{type(exc).__name__}: {str(exc)[:200]}")

    # 4. structured output with truncation repair off (how claim JSON will be parsed)
    print("\n== 4. Structured call, truncation repair off")
    try:
        provider = make(tracer=tracer)
        parsed, resp = provider.complete_json(
            CLAIMS_SYSTEM, CLAIMS_USER, role=Role.EXTRACTION, required_keys=["claims"],
            repair_truncation=False, use_cache=False,
        )
        ok = isinstance(parsed["claims"], list) and len(parsed["claims"]) >= 1
        record("structured", ok, f"{resp.model}: {parsed}")
    except Exception as exc:
        record("structured", False, f"{type(exc).__name__}: {str(exc)[:200]}")

    # 5. the trace accounts for the calls in checks 2-4
    print("\n== 5. Trace")
    tracer.close()
    totals = TraceReader.load(tracer.path).llm_totals()
    record("trace", totals["cached_calls"] >= 1 and totals["live_calls"] >= 3,
           f"{totals['calls']} calls ({totals['live_calls']} live, {totals['cached_calls']} cached), "
           f"{totals['total_tokens']} tokens")

    if not args.dry_run:
        print("\n== Quota ledger (" + str(resolve_path(llm.quota_ledger_path)) + ")")
        print(make().quota_report())

    failed = [name for name, ok, _ in results if not ok]
    print("\n== Result")
    print("SMOKE FAILED: " + ", ".join(failed) if failed else "SMOKE PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
