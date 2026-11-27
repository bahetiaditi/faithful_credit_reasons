import numpy as np
import polars as pl
import pytest

from fcr.config import RatioSpec
from fcr.features.screen import (
    OTHER_LEVEL,
    _rank_matrix,
    add_ratios,
    cap_categories,
    category_levels,
    correlation_screen,
    exclusion_status,
    missing_and_constant,
    recode_values,
)


def test_exclusion_names_the_reason():
    assert exclusion_status("sex_738L", {"sex_738L"}, {"zip_1M"}) == "excluded:protected"
    assert exclusion_status("zip_1M", {"sex_738L"}, {"zip_1M"}) == "excluded:proxy"
    assert exclusion_status("amount_1A", {"sex_738L"}, {"zip_1M"}) is None


def test_missing_and_constant_rules():
    df = pl.DataFrame({
        "sparse": [None] * 97 + [1.0, 2.0, 3.0],
        "flat": [1.0] * 100,
        "fine": list(np.arange(100, dtype=float)),
    })
    r = missing_and_constant(df, ["sparse", "flat", "fine"], max_missing=0.95, min_minority=5)
    assert r["sparse"].status == "dropped:missing" and r["sparse"].train_missing_share == pytest.approx(0.97)
    assert r["flat"].status == "dropped:constant" and r["flat"].top_share == 1.0
    assert r["fine"].status == "kept"


def test_correlation_drops_the_more_missing_of_a_pair():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    df = pl.DataFrame({"a": x, "b": x * 3 + 1, "c": -x, "d": rng.normal(size=2000)})
    df = df.with_columns(pl.when(pl.int_range(pl.len()) < 50).then(None).otherwise(pl.col("b")).alias("b"))
    missing = {"a": 0.0, "b": 0.025, "c": 0.0, "d": 0.0}
    r = correlation_screen(df, ["a", "b", "c", "d"], missing, 0.95)
    assert r["b"].status == "dropped:correlated" and r["b"].correlated_with == "a"
    assert r["c"].status == "dropped:correlated" and r["c"].rho == pytest.approx(-1.0)
    assert r["a"].status == "kept" and r["d"].status == "kept"


def test_unrelated_mostly_missing_columns_are_kept():
    rng = np.random.default_rng(1)
    n = 5000
    observed = rng.random(n) < 0.02
    a = np.where(observed, rng.normal(size=n), np.nan)
    b = np.where(observed, rng.normal(size=n), np.nan)
    df = pl.DataFrame({"a": a, "b": b}).fill_nan(None)
    r = correlation_screen(df, ["a", "b"], {"a": 0.98, "b": 0.98}, 0.95)
    assert r["a"].status == r["b"].status == "kept"


def test_a_fast_pass_flag_needs_exact_confirmation():
    """Identical on 20 shared rows and missing elsewhere: the filled fast pass
    sees rho near 1, but 20 co-observed rows are too few to confirm, so both stay."""
    values = [float(i) for i in range(20)] + [None] * 980
    df = pl.DataFrame({"a": values, "b": values})
    z = _rank_matrix(df, ["a", "b"])
    assert (z.T @ z / z.shape[0])[0, 1] > 0.95, "precondition: the fast pass flags the pair"
    r = correlation_screen(df, ["a", "b"], {"a": 0.98, "b": 0.98}, 0.95)
    assert r["a"].status == r["b"].status == "kept"


def test_category_levels_and_capping():
    train = pl.DataFrame({"k": ["x"] * 5 + ["y"] * 3 + ["z"] * 3 + ["w"] + [None] * 2})
    levels = category_levels(train, "k", 2)
    assert levels == ["x", "y"]                 # y and z tie at 3: alphabetical
    capped = cap_categories(pl.DataFrame({"k": ["x", "z", None, "new"]}), {"k": levels})
    assert capped["k"].to_list() == ["x", OTHER_LEVEL, None, OTHER_LEVEL]


def test_recode_only_touches_listed_values():
    df = pl.DataFrame({"inc": ["HANDICAPPED_2", "EMPLOYED", None, "HANDICAPPED"]})
    out = recode_values(df, "inc", ("HANDICAPPED", "HANDICAPPED_2"), "OTHER")
    assert out["inc"].to_list() == ["OTHER", "EMPLOYED", None, "OTHER"]


def test_ratios_guard_against_bad_denominators():
    df = pl.DataFrame({"num": [10.0, 5.0, None, 4.0], "den": [2.0, 0.0, 3.0, -1.0]})
    out, added = add_ratios(df, [RatioSpec(name="r", numerator="num", denominator="den"),
                                 RatioSpec(name="skipped", numerator="num", denominator="absent")])
    assert added == ["r"]
    assert out["r"].to_list() == [5.0, None, None, None]


def test_ties_go_to_the_more_explainable_feature():
    x = [float(i) for i in range(100)]
    df = pl.DataFrame({"actualdpd_1P__count": x, "applprev_1__rows": x})
    r = correlation_screen(df, list(df.columns), {c: 0.0 for c in df.columns}, 0.95,
                           priority={"applprev_1__rows": 0, "actualdpd_1P__count": 7})
    assert r["actualdpd_1P__count"].status == "dropped:correlated"
    assert r["actualdpd_1P__count"].correlated_with == "applprev_1__rows"


def test_a_feature_that_caused_a_drop_is_not_dropped_later():
    rng = np.random.default_rng(3)
    x = rng.normal(size=3000)
    # a~b strongest; b~c weaker; c is cleaner than b, so pairwise c would win.
    a = x
    b = x + rng.normal(scale=0.05, size=3000)
    c = b + rng.normal(scale=0.2, size=3000)
    df = pl.DataFrame({"a": a, "b": b, "c": c})
    missing = {"a": 0.01, "b": 0.0, "c": 0.0}
    r = correlation_screen(df, ["a", "b", "c"], missing, 0.95)
    kept = {f for f, rec in r.items() if rec.status == "kept"}
    for rec in r.values():
        if rec.status == "dropped:correlated":
            assert rec.correlated_with in kept


def test_rare_but_populated_values_survive_the_constant_rule():
    """0.4% non-zero is a 99.6% top share, but 400 of 100k rows is enough to learn from."""
    n = 100_000
    dpd = [0.0] * (n - 400) + [30.0] * 400
    df = pl.DataFrame({"dpd": dpd, "almost": [0.0] * (n - 50) + [1.0] * 50})
    r = missing_and_constant(df, ["dpd", "almost"], max_missing=0.95, min_minority=100)
    assert r["dpd"].status == "kept" and r["dpd"].minority_rows == 400
    assert r["dpd"].top_share == pytest.approx(0.996)
    assert r["almost"].status == "dropped:constant" and r["almost"].minority_rows == 50


def test_preferred_ratios_win_their_correlation_ties():
    rng = np.random.default_rng(4)
    debt = np.where(rng.random(3000) < 0.7, 0.0, rng.gamma(2.0, 1000.0, 3000))
    income = rng.uniform(900, 1100, 3000)
    df = pl.DataFrame({"currdebt_1A": debt, "debt_to_income__ratio": debt / income})
    missing = {"currdebt_1A": 0.0, "debt_to_income__ratio": 0.001}
    plain = correlation_screen(df, list(df.columns), missing, 0.95)
    assert plain["debt_to_income__ratio"].status == "dropped:correlated", "precondition: the raw amount wins on missing share"
    r = correlation_screen(df, list(df.columns), missing, 0.95, preferred={"debt_to_income__ratio"})
    assert r["debt_to_income__ratio"].status == "kept"
    assert r["currdebt_1A"].correlated_with == "debt_to_income__ratio"
