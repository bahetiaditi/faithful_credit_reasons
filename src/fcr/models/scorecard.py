"""S0: the WoE logistic-regression scorecard and its points-below-maximum reasons (C8).

The scorecard is the benchmark a bank's validation team already trusts, and its
reason codes are the classical reference the SHAP-based reasons are compared to
later. Built in three steps, all on training weeks only:

1. Bin every candidate feature (optbinning; numeric bins forced monotone so each
   feature's points move in one direction) and rank by information value (IV).
2. Choose up to `max_features` greedily by IV, skipping a feature whose WoE is
   more correlated than `max_woe_correlation` with one already chosen, and any
   feature outside [min_iv, max_iv]. Masked features are left out because a
   scorecard reason must be nameable.
3. Fit L2 logistic regression on WoE and scale to points (PDO). Higher WoE means
   lower risk, so every coefficient must be negative; a positive one means
   collinearity has flipped a sign, and that feature is removed and the model refit.

Reasons: for each feature, the points the applicant lost relative to the most
points any training applicant could get for it ("points below maximum"); the
largest losses, up to four, are the reasons.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl
from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression

from fcr.config import ScorecardConfig

TRANSFORM = {"metric_missing": "empirical", "metric_special": "empirical"}


def to_pandas(df: pl.DataFrame, features: list[str], categorical: set[str]) -> pd.DataFrame:
    """optbinning input: numbers as float64, categories as object strings with NaN for missing."""
    out = {}
    for f in features:
        s = df[f]
        if f in categorical:
            out[f] = pd.Series(s.cast(pl.String).to_list(), dtype=object)
        else:
            out[f] = s.cast(pl.Float64).to_numpy()
    return pd.DataFrame(out)


def _binning(features: list[str], categorical: set[str], cfg: ScorecardConfig) -> BinningProcess:
    return BinningProcess(
        variable_names=features,
        categorical_variables=[f for f in features if f in categorical],
        max_n_prebins=cfg.max_n_prebins,
        min_prebin_size=cfg.min_prebin_size,
        binning_fit_params={f: {"monotonic_trend": "auto_asc_desc"} for f in features if f not in categorical},
        binning_transform_params={f: dict(TRANSFORM) for f in features},
        n_jobs=-1,
    )


@dataclass
class Ranking:
    """Every candidate binned on training weeks, ranked by IV, with WoE on a row sample for correlations."""
    iv: dict[str, float]
    records: dict[str, dict]
    eligible: list[str]                 # passed the binning and IV rules, best first
    woe: dict[str, np.ndarray]          # eligible feature -> WoE on the correlation sample
    y: np.ndarray                       # target on the same sample rows


def rank_features(train: pl.DataFrame, candidates: list[str], categorical: set[str],
                  cfg: ScorecardConfig, seed: int) -> Ranking:
    y = train["target"].to_numpy()
    X = to_pandas(train, candidates, categorical)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bp = _binning(candidates, categorical, cfg).fit(X, y)
    summary = bp.summary().set_index("name")
    iv = {f: float(summary.loc[f, "iv"]) for f in candidates}
    status = {f: str(summary.loc[f, "status"]) for f in candidates}
    records = {f: {"feature": f, "iv": iv[f], "n_bins": int(summary.loc[f, "n_bins"]),
                   "binning_status": status[f], "selected": False, "reason": ""} for f in candidates}
    eligible = []
    for f in sorted(candidates, key=lambda f: (-iv[f], f)):
        if status[f] != "OPTIMAL":
            records[f]["reason"] = f"binning {status[f].lower()}"
        elif iv[f] < cfg.min_iv:
            records[f]["reason"] = f"iv below {cfg.min_iv}"
        elif iv[f] > cfg.max_iv:
            records[f]["reason"] = f"iv above {cfg.max_iv}: suspiciously strong, check for leakage"
        else:
            eligible.append(f)
    n = min(cfg.corr_sample_cases, train.height)
    rows = np.sort(np.random.default_rng(seed).choice(train.height, size=n, replace=False))
    woe_all = np.asarray(bp.transform(X.iloc[rows], metric="woe", **TRANSFORM))
    position = {f: j for j, f in enumerate(candidates)}
    woe = {f: woe_all[:, position[f]] for f in eligible}
    return Ranking(iv, records, eligible, woe, y[rows])


def _signs_ok(ranking: Ranking, features: list[str], c: float) -> tuple[bool, np.ndarray]:
    X = np.column_stack([ranking.woe[f] for f in features])
    coef = LogisticRegression(C=c, max_iter=1000).fit(X, ranking.y).coef_[0]
    return bool((coef < 0).all()), coef


def greedy_select(ranking: Ranking, cfg: ScorecardConfig, banned: set[str] = frozenset()) -> tuple[list[str], dict[str, str]]:
    """Forward selection by IV with two gates per candidate.

    1. Its WoE correlates no more than max_woe_correlation with any chosen feature.
    2. Refitting on the WoE sample with it added keeps every coefficient negative.
       A positive coefficient means collinearity has flipped a sign: the feature
       would then *add* points for worse behaviour, which no reason code can explain.
    """
    chosen: list[str] = []
    why: dict[str, str] = {}
    for f in ranking.eligible:
        if f in banned:
            continue
        if len(chosen) == cfg.max_features:
            why[f] = f"limit of {cfg.max_features} features reached"
            continue
        col = ranking.woe[f]
        if col.std() == 0:
            why[f] = "constant woe"
            continue
        worst = max(((abs(float(np.corrcoef(col, ranking.woe[c])[0, 1])), c) for c in chosen), default=(0.0, ""))
        if worst[0] > cfg.max_woe_correlation:
            why[f] = f"woe correlation {worst[0]:.2f} with {worst[1]}"
            continue
        ok, coef = _signs_ok(ranking, chosen + [f], cfg.regularization_c)
        if not ok:
            bad = [n for n, b in zip(chosen + [f], coef) if b >= 0]
            why[f] = f"adding it flips a coefficient sign ({', '.join(bad)})"
            continue
        chosen.append(f)
    return chosen, why


@dataclass
class S0:
    """A fitted scorecard: optbinning model plus the points table used for reasons."""
    scorecard: Scorecard
    features: list[str]
    categorical: set[str]
    table: pl.DataFrame            # feature, bin_id, bin, count, event_rate, woe, points
    max_points: dict[str, float]
    factor: float
    offset: float

    def _indices(self, df: pl.DataFrame) -> np.ndarray:
        X = to_pandas(df, self.features, self.categorical)
        return np.asarray(self.scorecard.binning_process_.transform(X, metric="indices", **TRANSFORM)).astype(int)

    def points(self, df: pl.DataFrame) -> np.ndarray:
        """Points per applicant per feature (n x k), looked up from the points table."""
        idx = self._indices(df)
        out = np.empty(idx.shape, dtype=float)
        for j, f in enumerate(self.features):
            lookup = dict(zip(*self.table.filter(pl.col("feature") == f).select("bin_id", "points").get_columns()))
            out[:, j] = [lookup[i] for i in idx[:, j]]
        return out

    def bins(self, df: pl.DataFrame) -> list[list[str]]:
        idx = self._indices(df)
        labels = {f: dict(zip(*self.table.filter(pl.col("feature") == f).select("bin_id", "bin").get_columns()))
                  for f in self.features}
        return [[labels[f][i] for f, i in zip(self.features, row)] for row in idx]

    def score(self, df: pl.DataFrame) -> np.ndarray:
        return self.points(df).sum(axis=1)

    def pd(self, df: pl.DataFrame) -> np.ndarray:
        """Probability of default implied by the score (good:bad odds = exp((score - offset) / factor))."""
        odds = np.exp((self.score(df) - self.offset) / self.factor)
        return 1.0 / (1.0 + odds)

    def reasons(self, df: pl.DataFrame, k: int = 4) -> list[list[tuple[str, float]]]:
        """Up to k (feature, points below maximum) per applicant, largest loss first, ties by name."""
        pts = self.points(df)
        gap = np.array([self.max_points[f] for f in self.features])[None, :] - pts
        names = np.array(self.features)
        order_names = np.argsort(names)           # stable tie-break by feature name
        out = []
        for row in gap:
            ranked = order_names[np.argsort(-row[order_names], kind="stable")]
            out.append([(str(names[j]), float(row[j])) for j in ranked[:k] if row[j] > 1e-9])
        return out


def fit_scorecard(train: pl.DataFrame, features: list[str], categorical: set[str], cfg: ScorecardConfig) -> S0:
    """One fit: bin the chosen features on training weeks, logistic regression on WoE, PDO scaling."""
    sc = Scorecard(
        _binning(features, categorical, cfg),
        LogisticRegression(C=cfg.regularization_c, max_iter=2000),
        scaling_method="pdo_odds",
        scaling_method_params={"pdo": cfg.pdo, "odds": cfg.base_odds, "scorecard_points": cfg.base_points},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.fit(to_pandas(train, features, categorical), train["target"].to_numpy())
    t = sc.table(style="detailed")
    table = pl.DataFrame({
        "feature": t["Variable"].astype(str).to_list(), "bin_id": t["Bin id"].astype(int).to_list(),
        "bin": t["Bin"].astype(str).to_list(), "count": t["Count"].astype(int).to_list(),
        "event_rate": t["Event rate"].astype(float).to_list(), "woe": t["WoE"].astype(float).to_list(),
        "iv": t["IV"].astype(float).to_list(), "coefficient": t["Coefficient"].astype(float).to_list(),
        "points": t["Points"].astype(float).to_list(),
    })
    # the most points any training applicant actually received (empty bins are not attainable)
    max_points = dict(table.filter(pl.col("count") > 0).group_by("feature").agg(pl.col("points").max()).iter_rows())
    factor = cfg.pdo / np.log(2)
    offset = cfg.base_points - factor * np.log(cfg.base_odds)
    return S0(sc, list(features), set(categorical), table, max_points, factor, offset)


def build_s0(train: pl.DataFrame, candidates: list[str], categorical: set[str], cfg: ScorecardConfig,
             seed: int, max_refits: int = 5) -> tuple[S0, list[dict]]:
    """Rank, select (sign-constrained on the WoE sample), then fit on all training rows. If the
    full fit still flips a sign, that feature is banned and selection re-runs (rare: the
    sample and the full data rarely disagree).

    Returns the model and one record per candidate (iv, selected, and why not).
    """
    ranking = rank_features(train, candidates, categorical, cfg, seed)
    banned: dict[str, float] = {}
    for _ in range(max_refits + 1):
        chosen, why = greedy_select(ranking, cfg, set(banned))
        if len(chosen) < 2:
            raise RuntimeError("fewer than two features survive selection")
        s0 = fit_scorecard(train, chosen, categorical & set(chosen), cfg)
        coef = s0.scorecard.estimator_.coef_[0]
        if (coef < 0).all():
            break
        j = int(np.argmax(coef))
        banned[chosen[j]] = float(coef[j])
    else:
        raise RuntimeError(f"coefficients still have wrong signs after {max_refits} refits")
    records = []
    for f in sorted(ranking.records, key=lambda f: (-ranking.iv[f], f)):
        r = dict(ranking.records[f])
        if f in chosen:
            r["selected"], r["reason"] = True, ""
        elif f in banned:
            r["reason"] = f"coefficient sign flipped ({banned[f]:+.3f}): collinear with chosen features"
        elif f in why:
            r["reason"] = why[f]
        records.append(r)
    return s0, records
