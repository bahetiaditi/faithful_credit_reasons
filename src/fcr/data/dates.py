"""Leakage control by date role (REFERENCE.md 5.1 step 3, rules in configs/date_roles.yaml).

Applied to one chunk at a time, after the chunk is joined to its cases'
date_decision. The order matters:

1. sentinel dates become null (they are placeholders, not dates);
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
from pydantic import BaseModel, ConfigDict

DATE_FORMAT = "%Y-%m-%d"
DECISION = "__decision_date"


class Sentinels(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    before: date
    after: date


class DateRoles(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sentinels: Sentinels
    scheduled: tuple[str, ...] = ()
    anchors: dict[str, tuple[str, ...]] = {}

    def is_scheduled(self, column: str) -> bool:
        return column in self.scheduled


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
                  "scheduled_future_kept": {}}

    # 1. parse and null sentinels
    parsed = df.with_columns([_as_date(df, c).alias(c) for c in cols])
    lo, hi = pl.lit(roles.sentinels.before), pl.lit(roles.sentinels.after)
    sentinel_counts = parsed.select([((pl.col(c) < lo) | (pl.col(c) >= hi)).sum().alias(c) for c in cols]).row(0, named=True) if cols else {}
    diag["sentinels_nulled"] = {c: n for c, n in sentinel_counts.items() if n}
    parsed = parsed.with_columns(
        [pl.when((pl.col(c) < lo) | (pl.col(c) >= hi)).then(None).otherwise(pl.col(c)).alias(c) for c in cols]
    )

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
