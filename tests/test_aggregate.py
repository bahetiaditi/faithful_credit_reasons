import polars as pl
import pytest

from fcr.data.aggregate import aggregate_chunk

STATS = ("mean", "min", "max", "count")


def depth1():
    return pl.DataFrame({
        "case_id": [1, 1, 1, 2],
        "num_group1": [0, 1, 2, 0],
        "amount_1A": [1.0, 3.0, None, 5.0],
        "dpd_2P": [0.0, 7.0, 2.0, 0.0],
        "kind_3M": ["x", "y", "y", None],
        "flag_4L": [True, False, None, True],
        "tie_5M": ["b", "a", None, "z"],
    })


@pytest.fixture
def agg():
    out, specs = aggregate_chunk(depth1(), "t_1", 1, STATS)
    return out.sort("case_id"), specs


def row(out, case_id):
    return out.filter(pl.col("case_id") == case_id).row(0, named=True)


def test_numeric_stats_match_hand_computation(agg):
    out, _ = agg
    r = row(out, 1)
    assert (r["amount_1A__mean"], r["amount_1A__min"], r["amount_1A__max"], r["amount_1A__count"]) == (2.0, 1.0, 3.0, 2)
    assert row(out, 2)["amount_1A__count"] == 1


def test_dpd_columns_count_positive_values(agg):
    out, _ = agg
    assert row(out, 1)["dpd_2P__count_positive"] == 2
    assert row(out, 2)["dpd_2P__count_positive"] == 0


def test_rows_per_case(agg):
    out, _ = agg
    assert out["t_1__rows"].to_list() == [3, 1]


def test_categories_get_mode_and_distinct_non_null_count(agg):
    out, _ = agg
    r1, r2 = row(out, 1), row(out, 2)
    assert r1["kind_3M__mode"] == "y" and r1["kind_3M__n_unique"] == 2
    assert r2["kind_3M__mode"] is None and r2["kind_3M__n_unique"] == 0, "nulls are not a category"


def test_mode_ties_are_broken_deterministically(agg):
    out, _ = agg
    assert row(out, 1)["tie_5M__mode"] == "a"   # a and b once each: alphabetical


def test_booleans_become_share_true(agg):
    out, _ = agg
    assert row(out, 1)["flag_4L__share_true"] == pytest.approx(0.5)
    assert row(out, 1)["flag_4L__count"] == 2


def test_manifest_lists_every_feature_with_its_source(agg):
    out, specs = agg
    assert [s.feature for s in specs] == [c for c in out.columns if c != "case_id"]
    by_feature = {s.feature: s for s in specs}
    assert by_feature["amount_1A__max"].source == "amount_1A"
    assert by_feature["t_1__rows"].source == "" and by_feature["kind_3M__mode"].kind == "category"


def test_floats_are_stored_as_float32(agg):
    out, _ = agg
    assert out.schema["amount_1A__mean"] == pl.Float32


def test_depth0_passes_through():
    df = pl.DataFrame({"case_id": [1, 2], "x_1A": [1.0, 2.0], "k_2M": ["a", None]})
    out, specs = aggregate_chunk(df, "s_0", 0, STATS)
    assert out.equals(df) and [s.stat for s in specs] == ["value", "value"]


def test_applicant_row_is_not_averaged_with_relatives():
    df = pl.DataFrame({"case_id": [1, 1, 1, 2], "num_group1": [0, 1, 2, 0],
                       "income_1A": [100.0, 999.0, 999.0, 50.0], "role_2L": ["self", "spouse", "child", "self"]})
    out, specs = aggregate_chunk(df, "person_1", 1, STATS, applicant_row=True)
    out = out.sort("case_id")
    assert out["income_1A__applicant"].to_list() == [100.0, 50.0]
    assert out["person_1__related_persons"].to_list() == [2, 0]
    assert {s.stat for s in specs} == {"applicant", "related_persons"}


def test_depth2_counts_distinct_groups():
    df = pl.DataFrame({"case_id": [1, 1, 1], "num_group1": [0, 0, 1], "num_group2": [0, 1, 0], "x_1A": [1.0, 2.0, 3.0]})
    out, _ = aggregate_chunk(df, "t_2", 2, STATS)
    assert (out["t_2__rows"][0], out["t_2__groups"][0]) == (3, 2)
