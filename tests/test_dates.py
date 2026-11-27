import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from fcr.config import load_config, resolve_path
from fcr.data.dates import DECISION, DateRoles, apply_date_rules, attach_decision_dates, load_date_roles

ROLES = DateRoles.model_validate({
    "sentinels": {"before": "1901-01-01", "after": "2090-01-01"},
    "scheduled": ["end_3D"],
    "anchors": {"t_1": ["created_1D", "created_alt_9D"]},
})
DECISIONS = pl.DataFrame({"case_id": [1, 2], "date_decision": [date(2020, 1, 10), date(2020, 1, 20)]})


def chunk():
    return pl.DataFrame({
        "case_id": [1, 1, 1, 2, 2],
        "num_group1": [0, 1, 2, 0, 1],
        #             before,       AFTER (drop), null->alt,  before,       null both
        "created_1D": ["2020-01-01", "2020-01-15", None, "2019-12-01", None],
        "created_alt_9D": [None, None, "2020-01-05", None, None],
        #             before,       -,            AFTER,      sentinel,     before
        "paid_2D": ["2020-01-05", "2020-01-16", "2020-02-01", "1900-01-15", "2020-01-19"],
        #             future ok,    -,            sentinel,   past,         future ok
        "end_3D": ["2021-01-10", "2021-01-15", "2098-01-15", "2019-01-20", "2020-03-20"],
        "amount_4A": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def ruled():
    return apply_date_rules(attach_decision_dates(chunk(), DECISIONS), "t_1", ROLES)


def test_records_created_after_the_decision_are_dropped(ruled):
    df, diag = ruled
    assert df["amount_4A"].to_list() == [1.0, 3.0, 4.0, 5.0]   # row with amount 2.0 was created after its decision
    assert diag["rows_dropped_by_anchor"] == 1
    assert diag["rows_without_anchor"] == 1                    # case 2's second row: kept, cannot be judged


def test_anchor_columns_are_coalesced_in_order(ruled):
    df, _ = ruled
    # amount 3.0 has no created_1D; created_alt_9D (2020-01-05) is before its decision, so it stays.
    assert 3.0 in df["amount_4A"].to_list()


def test_future_event_values_become_null_and_scheduled_ones_stay(ruled):
    df, diag = ruled
    paid = dict(zip(df["amount_4A"].to_list(), df["paid_2D"].to_list()))
    end = dict(zip(df["amount_4A"].to_list(), df["end_3D"].to_list()))
    assert paid[3.0] is None                    # 2020-02-01 is after 2020-01-10
    assert paid[1.0] == 5                       # 2020-01-10 minus 2020-01-05
    assert end[1.0] == -366                     # scheduled a year ahead: kept, negative
    assert diag["future_events_nulled"] == {"paid_2D": 1}
    assert diag["scheduled_future_kept"] == {"end_3D": 2}


def test_sentinels_become_null_before_anything_else(ruled):
    df, diag = ruled
    paid = dict(zip(df["amount_4A"].to_list(), df["paid_2D"].to_list()))
    end = dict(zip(df["amount_4A"].to_list(), df["end_3D"].to_list()))
    assert paid[4.0] is None and end[3.0] is None
    assert diag["sentinels_nulled"] == {"paid_2D": 1, "end_3D": 1}


def test_no_event_date_is_after_its_decision(ruled):
    df, _ = ruled
    for c in ("created_1D", "created_alt_9D", "paid_2D"):
        assert (df[c].drop_nulls() >= 0).all(), c


def test_dates_are_integer_days(ruled):
    df, _ = ruled
    assert df.schema["paid_2D"] == pl.Int32 and df.schema[DECISION] == pl.Date


def test_missing_anchor_column_is_a_config_error():
    df = attach_decision_dates(chunk().drop("created_alt_9D"), DECISIONS)
    with pytest.raises(ValueError, match="anchor columns"):
        apply_date_rules(df, "t_1", ROLES)


def test_rules_require_decision_dates():
    with pytest.raises(ValueError, match="attach_decision_dates"):
        apply_date_rules(chunk(), "t_1", ROLES)


# --- the repo's date roles against the real schema -------------------------

C4_REGISTRY = Path(__file__).resolve().parents[1] / "results" / "c04" / "schema_registry.json"


def test_repo_date_roles_name_real_date_columns(repo_root):
    if not C4_REGISTRY.is_file():
        pytest.skip("results/c04 not present")
    cfg = load_config(repo_root / "configs" / "config.yaml")
    roles = load_date_roles(resolve_path(cfg.features.date_roles_file, repo_root))
    families = json.loads(C4_REGISTRY.read_text())["families"]
    date_cols = {c["name"] for f in families.values() for c in f["columns"] if c["name"].endswith("D")}
    assert set(roles.scheduled) <= date_cols
    for family, anchors in roles.anchors.items():
        family_cols = {c["name"] for c in families[family]["columns"]}
        assert set(anchors) <= family_cols, family
        assert all(a.endswith("D") for a in anchors)
    assert not set(roles.scheduled) & {a for anchors in roles.anchors.values() for a in anchors}, \
        "an anchor is when a record began, so it cannot be a scheduled date"
