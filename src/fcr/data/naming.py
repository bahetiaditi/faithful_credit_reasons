"""The competition's naming conventions, in one place.

File names: ``{split}_{family}_{depth}[_{chunk}]``, for example
``train_static_0_0`` (depth 0, chunk 0), ``train_static_cb_0`` (depth 0, one
file), ``train_credit_bureau_a_2_10`` (depth 2, chunk 10). ``train_base`` is
the label table and has no depth suffix. Chunks of one family share the same
columns and must be concatenated by rows.

Column names end in a transformation code after the last underscore-number:
P days past due, A amount, D date, M masked categorical, T and L unspecified.
Key columns (case_id, num_group1, ...) carry no code.

The synthetic fixtures and the C3 loader both rely on this module, so the
fixtures cannot drift from what the loader expects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Split = Literal["train", "test"]

KEY_COLUMNS: frozenset[str] = frozenset(
    {"case_id", "num_group1", "num_group2", "date_decision", "MONTH", "WEEK_NUM", "target"}
)
FEATURE_SUFFIXES: frozenset[str] = frozenset("PAMDTL")
GROUP_COLUMNS_BY_DEPTH: dict[int, tuple[str, ...]] = {
    0: (),
    1: ("num_group1",),
    2: ("num_group1", "num_group2"),
}

# Lazy family match so "static_cb_0" parses as family "static_cb", depth 0,
# rather than family "static" followed by junk.
_TABLE_RE = re.compile(
    r"^(?P<split>train|test)_(?P<family>[a-z][a-z_]*?)_(?P<depth>[0-2])(?:_(?P<chunk>\d+))?$"
)
_BASE_RE = re.compile(r"^(?P<split>train|test)_base$")
_FEATURE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*_\d+(?P<suffix>[A-Z])$")


@dataclass(frozen=True)
class TableName:
    split: Split
    family: str
    depth: int
    chunk: int | None

    @property
    def is_base(self) -> bool:
        return self.family == "base"

    def stem(self) -> str:
        if self.is_base:
            return f"{self.split}_base"
        chunk = "" if self.chunk is None else f"_{self.chunk}"
        return f"{self.split}_{self.family}_{self.depth}{chunk}"


def parse_table_name(stem: str) -> TableName:
    """Parse a file stem (no extension) into split, family, depth and chunk."""
    base = _BASE_RE.match(stem)
    if base:
        return TableName(split=base["split"], family="base", depth=0, chunk=None)  # type: ignore[arg-type]
    m = _TABLE_RE.match(stem)
    if not m:
        raise ValueError(f"not a competition table name: {stem!r}")
    chunk = int(m["chunk"]) if m["chunk"] is not None else None
    return TableName(split=m["split"], family=m["family"], depth=int(m["depth"]), chunk=chunk)  # type: ignore[arg-type]


def feature_suffix(column: str) -> str | None:
    """Return the transformation code of a feature column, or None for key columns.

    Raises if a non-key column does not follow the convention, so a schema
    surprise in the real data fails loudly instead of being silently miscast.
    """
    if column in KEY_COLUMNS:
        return None
    m = _FEATURE_RE.match(column)
    if not m or m["suffix"] not in FEATURE_SUFFIXES:
        raise ValueError(f"column does not follow the suffix convention: {column!r}")
    return m["suffix"]
