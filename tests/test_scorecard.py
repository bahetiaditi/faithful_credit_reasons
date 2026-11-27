"""C8: S0 scorecard selection, signs, points and points-below-maximum reasons."""

import numpy as np
import polars as pl
import pytest

from fcr.config import load_config
from fcr.eval.metrics import discrimination
from fcr.models.guard import HeldOutWeeksTouched, assert_no_test, development_table
from fcr.models.scorecard import build_s0, greedy_select, rank_features, to_pandas

N = 20000


@pytest.fixture(scope="module")
def scfg(repo_root):
    cfg = load_config(repo_root / "configs" / "config.yaml").model.scorecard
    return cfg.model_copy(update={"max_features": 5, "corr_sample_cases": N})


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    dpd = rng.exponential(10, N) * (rng.random(N) < 0.4)          # more days past due: riskier
    income = rng.lognormal(10, 0.5, N)                            # more income: safer
    status = rng.choice(["A", "B", "C"], N, p=[0.6, 0.3, 0.1])    # C riskiest
    logit = -3.4 + 0.035 * dpd - 0.9 * (np.log(income) - 10) + np.select([status == "B", status == "C"], [0.5, 1.2], 0)
    target = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    dpd_copy = dpd * 1.01 + 0.1                                    # a near-duplicate of dpd
    noise = rng.normal(size=N)
    income_with_gaps = np.where(rng.random(N) < 0.1, np.nan, income)
    return pl.DataFrame({
        "case_id": np.arange(N), "WEEK_NUM": rng.integers(0, 64, N), "target": target,
        "dpd_1P": dpd, "dpdcopy_2P": dpd_copy, "income_3A": income_with_gaps,
        "status_4L": status, "noise_5L": noise,
    }).with_columns(pl.col("income_3A").fill_nan(None))


FEATURES = ["dpd_1P", "dpdcopy_2P", "income_3A", "status_4L", "noise_5L"]
CATS = {"status_4L"}


@pytest.fixture(scope="module")
def fitted(data, scfg):
    return build_s0(data, FEATURES, CATS, scfg, seed=1)


def test_selection_drops_noise_and_the_duplicate(fitted):
    s0, records = fitted
    by = {r["feature"]: r for r in records}
    assert set(s0.features) == {"dpd_1P", "income_3A", "status_4L"} or set(s0.features) == {"dpdcopy_2P", "income_3A", "status_4L"}
    dup = "dpdcopy_2P" if "dpd_1P" in s0.features else "dpd_1P"
    assert by[dup]["reason"].startswith("woe correlation")
    assert by["noise_5L"]["reason"].startswith("iv below")
    assert [r["iv"] for r in records] == sorted((r["iv"] for r in records), reverse=True)


def test_every_coefficient_is_negative(fitted):
    s0, _ = fitted
    assert (s0.scorecard.estimator_.coef_[0] < 0).all()


def test_points_sum_to_optbinning_score_and_scale_as_configured(fitted, data, scfg):
    s0, _ = fitted
    native = np.asarray(s0.scorecard.score(to_pandas(data, s0.features, s0.categorical)))
    assert np.abs(s0.score(data) - native).max() < 1e-6
    # base_points at base_odds, pdo points per doubling
    pd_at = lambda score: 1 / (1 + np.exp((score - s0.offset) / s0.factor))
    assert pd_at(scfg.base_points) == pytest.approx(1 / (1 + scfg.base_odds))
    assert pd_at(scfg.base_points + scfg.pdo) == pytest.approx(1 / (1 + 2 * scfg.base_odds))
    order = np.argsort(s0.score(data))
    assert np.all(np.diff(s0.pd(data)[order]) <= 1e-12), "PD must fall as the score rises"


def test_discrimination_is_real(fitted, data):
    s0, _ = fitted
    assert discrimination(data["target"].to_numpy(), -s0.score(data))["roc_auc"] > 0.65


def test_reasons_are_points_below_the_attainable_maximum(fitted, data):
    s0, _ = fitted
    pts = s0.points(data)
    for i, rs in enumerate(s0.reasons(data.head(300), k=4)):
        assert len(rs) <= 4
        gaps = [g for _, g in rs]
        assert gaps == sorted(gaps, reverse=True) and all(g > 0 for g in gaps)
        for f, g in rs:
            assert g == pytest.approx(s0.max_points[f] - pts[i, s0.features.index(f)])
    attainable = s0.table.filter(pl.col("count") > 0)
    for f, mx in s0.max_points.items():
        assert mx == attainable.filter(pl.col("feature") == f)["points"].max()


def test_an_applicant_at_every_maximum_gets_no_reasons(fitted, data):
    s0, _ = fitted
    pts = s0.points(data)
    best = np.isclose(pts, np.array([s0.max_points[f] for f in s0.features])).all(axis=1)
    if best.any():
        assert s0.reasons(data.filter(pl.Series(best)).head(1))[0] == []


def test_sign_constraint_rejects_a_feature_that_would_flip(data, scfg):
    # a feature that is mostly a copy of income but kept below the correlation limit by noise:
    # adding it beside income would split income's effect and can flip a sign; the gate must
    # either reject it or keep every coefficient negative
    rng = np.random.default_rng(5)
    df = data.with_columns((pl.col("income_3A") * pl.Series(rng.lognormal(0, 0.6, N))).alias("incomeish_6A"))
    ranking = rank_features(df, FEATURES + ["incomeish_6A"], CATS, scfg.model_copy(update={"max_woe_correlation": 0.95}), 1)
    chosen, why = greedy_select(ranking, scfg.model_copy(update={"max_woe_correlation": 0.95}))
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([ranking.woe[f] for f in chosen])
    assert (LogisticRegression(C=scfg.regularization_c, max_iter=1000).fit(X, ranking.y).coef_[0] < 0).all()


def test_the_guard_rejects_test_weeks(repo_root, data):
    cfg = load_config(repo_root / "configs" / "config.yaml")
    assert_no_test(data, cfg)
    with pytest.raises(HeldOutWeeksTouched):
        assert_no_test(data.with_columns(pl.lit(80).alias("WEEK_NUM")), cfg)
    with pytest.raises(HeldOutWeeksTouched):
        assert_no_test(data.with_columns(pl.lit("test").alias("split")), cfg)


def test_development_table_removes_test_rows(repo_root):
    from fcr.data.dataset import FeatureTable
    from fcr.data.split import SPLIT_DTYPE
    cfg = load_config(repo_root / "configs" / "config.yaml")
    m = pl.DataFrame({"case_id": [1, 2, 3], "WEEK_NUM": [10, 70, 85], "target": [0, 1, 0],
                      "split": pl.Series(["train", "validation", "test"], dtype=SPLIT_DTYPE)})
    t = FeatureTable(m, pl.DataFrame({"case_id": [1, 2, 3], "sex": ["F", "M", "F"]}), {"features": []}, None, False)
    dev = development_table(t, cfg)
    assert dev.matrix["case_id"].to_list() == [1, 2] and dev.slices["case_id"].to_list() == [1, 2]


def test_discrimination_metrics():
    y = np.array([0, 0, 1, 1])
    m = discrimination(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1.0 and m["gini"] == 1.0 and m["ks"] == 1.0 and m["defaults"] == 2
    with pytest.raises(ValueError):
        discrimination(np.zeros(3), np.arange(3))
