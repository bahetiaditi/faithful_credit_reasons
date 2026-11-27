"""Leakage control by date role (REFERENCE.md 5.1 step 3, rules in configs/date_roles.yaml).

Applied to one chunk at a time, after the chunk is joined to its cases'
date_decision. The order matters:

1. sentinel dates become null (they are placeholders, not dates); every
   date's lag behind the decision is profiled; retrieval timestamps are then
   dropped (they record when data was pulled, not an applicant event);
2. records whose anchor date is after date_decision are dropped (the record
   did not exist when the decision was made);
3. remaining event dates after date_decision become null (future information
   about a record that did exist);
4. every date becomes days before date_decision (negative only for scheduled
   dates, which may lie in the future).

After this, no event-date value in the output is negative. That invariant is
what the C5 leakage test asserts.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

DATE_FORMAT = "%Y-%m-%d"
DECISION = "__decision_date"

# Lag of a date behind its case's decision, in days, as histogram buckets.
# "before" is on or before the decision; the rest are after it. Counts merge
# exactly across chunks, unlike quantiles.
LAG_BUCKETS = (("before", None, 0), ("1d", 1, 1), ("2-7d", 2, 7), ("8-30d", 8, 30),
               ("31-90d", 31, 90), ("91-365d", 91, 365), (">365d", 366, None))


class Sentinels(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    before: date
    after: date


class DateRoles(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sentinels: Sentinels
    scheduled: tuple[str, ...] = ()
    retrieval: tuple[str, ...] = ()
    anchors: dict[str, tuple[str, ...]] = {}
    # Reviewed anchor-drop shares: the C5 smoke warns only above max(0.20, this + 0.05).
    anchor_drop_expected: dict[str, float] = {}

    @model_validator(mode="after")
    def _roles_are_disjoint(self) -> DateRoles:
        anchors = {c for cols in self.anchors.values() for c in cols}
        for name, overlap in (("scheduled and retrieval", set(self.scheduled) & set(self.retrieval)),
                              ("anchor and retrieval", anchors & set(self.retrieval)),
                              ("anchor and scheduled", anchors & set(self.scheduled))):
            if overlap:
                raise ValueError(f"columns cannot be both {name}: {sorted(overlap)}")
        unknown = set(self.anchor_drop_expected) - set(self.anchors)
        if unknown:
            raise ValueError(f"anchor_drop_expected for families without an anchor: {sorted(unknown)}")
        if any(not 0.0 <= v < 1.0 for v in self.anchor_drop_expected.values()):
            raise ValueError("anchor_drop_expected shares must be in [0, 1)")
        return self

    def anchor_drop_limit(self, family: str, default: float = 0.20) -> float:
        expected = self.anchor_drop_expected.get(family)
        return default if expected is None else max(default, expected + 0.05)

    def is_scheduled(self, column: str) -> bool:
        return column in self.scheduled

    def role(self, column: str) -> str:
        if column in self.retrieval:
            return "retrieval"
        return "scheduled" if column in self.scheduled else "event"


def load_date_roles(path: str | Path) -> DateRoles:
    return DateRoles.model_validate(yaml.safe_load(Path(path).read_text()))


def date_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("D") and c != DECISION and c != "date_decision"]


def _as_date(df: pl.DataFrame, column: str) -> pl.Expr:
    dtype = df.schema[column]
    if dtype == pl.Date:
        return pl.col(column)
    if dtype == pl.String:
        return pl.col(column).str.to_date(DATE_FORMAT, strict=False)
    return pl.col(column).cast(pl.Date)


def attach_decision_dates(df: pl.DataFrame, decisions: pl.DataFrame) -> pl.DataFrame:
    """Join each row to its case's date_decision (decisions: case_id, date_decision as Date)."""
    return df.join(decisions.rename({"date_decision": DECISION}), on="case_id", how="left")


def apply_date_rules(df: pl.DataFrame, family_key: str, roles: DateRoles) -> tuple[pl.DataFrame, dict]:
    """Apply steps 1-4 to one chunk that already carries DECISION. Returns the frame and counts."""
    cols = date_columns(df)
    if DECISION not in df.columns:
        raise ValueError("attach_decision_dates must run before apply_date_rules")
    diag: dict = {"family": family_key, "rows_in": df.height, "sentinels_nulled": {}, "future_events_nulled": {},
                  "scheduled_future_kept": {}, "lag_profile": {}}

    # 1. parse and null sentinels
    parsed = df.with_columns([_as_date(df, c).alias(c) for c in cols])
    lo, hi = pl.lit(roles.sentinels.before), pl.lit(roles.sentinels.after)
    sentinel_counts = parsed.select([((pl.col(c) < lo) | (pl.col(c) >= hi)).sum().alias(c) for c in cols]).row(0, named=True) if cols else {}
    diag["sentinels_nulled"] = {c: n for c, n in sentinel_counts.items() if n}
    parsed = parsed.with_columns(
        [pl.when((pl.col(c) < lo) | (pl.col(c) >= hi)).then(None).otherwise(pl.col(c)).alias(c) for c in cols]
    )
    diag["lag_profile"] = lag_profile(parsed, cols)

    # 1b. retrieval timestamps are not features
    retrieval = [c for c in cols if c in roles.retrieval]
    parsed = parsed.drop(retrieval)
    diag["retrieval_dropped"] = retrieval
    cols = [c for c in cols if c not in roles.retrieval]

    # 2. drop records created after the decision
    anchors = [c for c in roles.anchors.get(family_key, ()) if c in parsed.columns]
    missing = [c for c in roles.anchors.get(family_key, ()) if c not in parsed.columns]
    if missing:
        raise ValueError(f"{family_key}: anchor columns {missing} not in the table")
    if anchors:
        anchor = pl.coalesce([pl.col(c) for c in anchors])
        keep = anchor.is_null() | (anchor <= pl.col(DECISION))
        diag["rows_dropped_by_anchor"] = int(parsed.select((~keep).sum()).item())
        diag["rows_without_anchor"] = int(parsed.select(anchor.is_null().sum()).item())
        parsed = parsed.filter(keep)
    else:
        diag["rows_dropped_by_anchor"] = 0

    # 3. null future event values; count, but keep, future scheduled values
    future = parsed.select([(pl.col(c) > pl.col(DECISION)).sum().alias(c) for c in cols]).row(0, named=True) if cols else {}
    for c, n in future.items():
        if n:
            (diag["scheduled_future_kept"] if roles.is_scheduled(c) else diag["future_events_nulled"])[c] = n
    parsed = parsed.with_columns(
        [pl.when(pl.col(c) > pl.col(DECISION)).then(None).otherwise(pl.col(c)).alias(c)
         for c in cols if not roles.is_scheduled(c)]
    )

    # 4. days before the decision
    parsed = parsed.with_columns([(pl.col(DECISION) - pl.col(c)).dt.total_days().cast(pl.Int32).alias(c) for c in cols])
    diag["rows_out"] = parsed.height
    return parsed, diag


def lag_profile(df: pl.DataFrame, cols: list[str]) -> dict[str, dict[str, int]]:
    """Per date column: non-null count, lag buckets, and how often it is constant within a case.

    A retrieval timestamp is typically a few days after the decision and the
    same for every row of a case; an event history is spread out before it.
    """
    if not cols:
        return {}
    exprs = []
    for c in cols:
        lag = (pl.col(c) - pl.col(DECISION)).dt.total_days()
        exprs.append(pl.col(c).is_not_null().sum().alias(f"{c}|non_null"))
        for name, lo, hi in LAG_BUCKETS:
            cond = lag.is_not_null()
            if lo is not None:
                cond = cond & (lag >= lo)
            if hi is not None:
                cond = cond & (lag <= hi)
            exprs.append(cond.sum().alias(f"{c}|{name}"))
    row = df.select(exprs).row(0, named=True)
    out: dict[str, dict[str, int]] = {}
    for c in cols:
        out[c] = {k.split("|", 1)[1]: int(v) for k, v in row.items() if k.startswith(f"{c}|")}
    if "num_group1" in df.columns:
        per_case = df.group_by("case_id").agg(
            [pl.col(c).drop_nulls().n_unique().alias(f"{c}|u") for c in cols]
            + [pl.col(c).count().alias(f"{c}|n") for c in cols]
        )
        for c in cols:
            multi = per_case.filter(pl.col(f"{c}|n") >= 2)
            out[c]["cases_multi"] = multi.height
            out[c]["cases_constant"] = int((multi[f"{c}|u"] == 1).sum())
    return out
