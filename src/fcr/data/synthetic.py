"""Synthetic data that mimics the competition schema.

Purpose: every commit's tests run on a clean clone with no competition data,
so the code is exercised on files that look like the real ones: the same
directory layout, file naming, depths and group indices, column-suffix codes,
date strings, masked hash categories, a near-empty column, protected
attributes, and a planted leakage case.

Version 2 (C4.1) aligns two quirks with what C4 measured on the real data:
missing categories are nulls, not empty strings (the real parquet has none),
and chunk files split at case boundaries, so no case spans two chunks (true of
every real chunked family, and what makes per-chunk aggregation exact).

Version 3 (C5) adds what the feature build needs to be tested against: a
creation date on previous applications (the anchor that drops records created
after the decision), and the three tax registries under their real column
names, each covering a successive period as in the real data, with a few
records dated after the decision.

What it is not: a realistic simulation of credit risk. The target has a known,
simple dependence on a few latent drivers so that later models can be checked
for learning *something*; no number measured on these files means anything
about the real data.

Fixture contract (asserted in tests/test_synthetic.py):
  - Tables are written under ``<root>/parquet_files/train/`` with real-style
    names, including chunked families whose chunks share identical columns.
  - Every non-key column follows the suffix convention in fcr.data.naming.
  - ``<root>/feature_definitions.csv`` has columns Variable, Description and
    covers every feature column.
  - ``applprev_1`` contains rows whose ``approvaldate_319D`` falls *after* the
    case's ``date_decision``; their count is recorded in the manifest so the
    C5 leakage filter can be tested against a known answer.
  - No case_id appears in more than one chunk of a family.
  - Output is a pure function of the spec: same seed, same bytes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

GENERATOR_VERSION = 3

# Real column names, taken from the public ETL write-up cited in REFERENCE.md
# Section 4. Everything else is invented and marked [synthetic] in the
# feature definitions.
REAL_PERSON_COLUMNS = (
    "birth_259D",
    "education_927M",
    "empl_employedtotal_800L",
    "empl_industry_691L",
    "familystate_447L",
    "incometype_1044T",
    "mainoccupationinc_384A",
    "sex_738L",
)
REAL_APPLPREV_COLUMNS = (
    "actualdpd_943P",
    "annuity_853A",
    "approvaldate_319D",
    "cancelreason_3545846M",
    "byoccupationinc_3656910L",
)

_EDUCATION_HASHES = ("a55475b1", "P97_36_170", "P33_146_175", "P106_81_188", "P17_36_170")
_EDUCATION_P = (0.52, 0.27, 0.17, 0.035, 0.005)
_EMPLOYED_TOTAL = (None, "MORE_FIVE", "MORE_ONE", "LESS_ONE")
_EMPLOYED_TOTAL_P = (0.65, 0.24, 0.09, 0.02)
_INDUSTRY = (None, "OTHER", "GOVERNMENT", "EDUCATION", "TRADE", "HEALTH")
_INDUSTRY_P = (0.66, 0.25, 0.03, 0.02, 0.02, 0.02)
_FAMILY = (None, "MARRIED", "SINGLE", "WIDOWED", "DIVORCED", "LIVING_WITH_PARTNER")
_FAMILY_P = (0.52, 0.31, 0.12, 0.025, 0.015, 0.01)
_SEX = (None, "F", "M")
_SEX_P = (0.3, 0.38, 0.32)
_INCOME_TYPE = (
    "PRIVATE_SECTOR_EMPLOYEE",
    "SALARIED_GOVT",
    "RETIRED_PENSIONER",
    "EMPLOYED",
    "SELFEMPLOYED",
    "OTHER",
    "HANDICAPPED_2",
    "HANDICAPPED_3",
)
_INCOME_TYPE_P = (0.32, 0.245, 0.2, 0.195, 0.02, 0.008, 0.007, 0.005)
_CLIENT_TYPE_HASHES = ("a55475b1", "c7a5ad39", "P24_27_36")
_CANCEL_HASHES = ("a55475b1", "P94_109_143", "P85_114_140")
_EMPLOYER_HASHES = ("f980a1ea", "5c3fa4a2", "0e46f5d5", "a55475b1")


@dataclass(frozen=True)
class SyntheticSpec:
    n_cases: int = 2000
    default_rate: float = 0.05
    n_weeks: int = 92
    seed: int = 0
    leak_share: float = 0.05            # share of applprev rows dated after the decision
    riskassessment_missing_share: float = 0.965
    drift_per_week: float = 0.005       # mild upward risk drift, so monitoring has a signal
    start_date: str = "2019-01-01"


@dataclass(frozen=True)
class SyntheticDataset:
    root: Path
    train_dir: Path
    manifest: dict = field(repr=False)


def _iso(d: date) -> str:
    return d.isoformat()


def _calibrated_intercept(score: np.ndarray, target_rate: float) -> float:
    """Find b so that mean(sigmoid(score + b)) == target_rate, by bisection."""
    lo, hi = -30.0, 30.0
    for _ in range(100):
        mid = (lo + hi) / 2
        rate = float(np.mean(1.0 / (1.0 + np.exp(-(score + mid)))))
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _write_chunks(df: pl.DataFrame, train_dir: Path, family: str, depth: int, n_chunks: int) -> list[str]:
    """Split into chunk files with identical columns, cutting only between cases.

    Rows are generated grouped by case, so each cut moves forward to the next
    row whose case_id differs from the row before it.
    """
    stems = []
    ids = df["case_id"].to_numpy()
    bounds = [0]
    for target in np.linspace(0, df.height, n_chunks + 1).astype(int)[1:-1]:
        cut = max(int(target), bounds[-1])
        while 0 < cut < df.height and ids[cut] == ids[cut - 1]:
            cut += 1
        bounds.append(cut)
    bounds.append(df.height)
    for i in range(n_chunks):
        stem = f"train_{family}_{depth}_{i}"
        df.slice(bounds[i], bounds[i + 1] - bounds[i]).write_parquet(train_dir / f"{stem}.parquet")
        stems.append(stem)
    return stems


def generate_synthetic_dataset(root: str | Path, spec: SyntheticSpec | None = None) -> SyntheticDataset:
    """Write a synthetic competition-shaped dataset under ``root`` and return its manifest."""
    spec = spec or SyntheticSpec()
    if not 0 < spec.default_rate < 1:
        raise ValueError("default_rate must be in (0, 1)")
    rng = np.random.default_rng(spec.seed)
    root = Path(root)
    train_dir = root / "parquet_files" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    n = spec.n_cases
    start = date.fromisoformat(spec.start_date)

    # ---- latent applicant drivers (the only things the target depends on) ----
    week = rng.integers(0, spec.n_weeks, size=n)
    decision = [start + timedelta(days=int(7 * w + rng.integers(0, 7))) for w in week]
    age_years = rng.uniform(21, 75, size=n)
    income = np.round(rng.lognormal(mean=np.log(50_000), sigma=0.5, size=n), -2)
    debt_ratio = np.clip(rng.beta(2, 5, size=n) * 1.5, 0, 1.5)
    has_dpd = rng.random(n) < 0.15
    worst_dpd = np.where(has_dpd, rng.integers(1, 120, size=n), 0)
    score = (
        1.4 * has_dpd
        + 0.012 * worst_dpd
        + 1.1 * debt_ratio
        - 0.6 * np.log(income / 50_000)
        - 0.012 * (age_years - 45)
        + spec.drift_per_week * week
    )
    p_default = 1.0 / (1.0 + np.exp(-(score + _calibrated_intercept(score, spec.default_rate))))
    target = (rng.random(n) < p_default).astype(np.int64)

    # Real case_ids are not contiguous; mimic that.
    case_id = np.sort(rng.choice(np.arange(n * 3), size=n, replace=False)).astype(np.int64)
    definitions: dict[str, str] = {}
    tables: dict[str, int] = {}

    def record(stem: str, df: pl.DataFrame) -> None:
        tables[stem] = df.height

    # ---- base ----
    base = pl.DataFrame(
        {
            "case_id": case_id,
            "date_decision": [_iso(d) for d in decision],
            "MONTH": np.array([d.year * 100 + d.month for d in decision], dtype=np.int64),
            "WEEK_NUM": week.astype(np.int64),
            "target": target,
        }
    )
    base.write_parquet(train_dir / "train_base.parquet")
    record("train_base", base)

    # ---- static_0: depth 0, two row chunks ----
    credit_amount = np.round(income * rng.uniform(0.5, 4.0, size=n), -2)
    static0 = pl.DataFrame(
        {
            "case_id": case_id,
            "syn_credamount_1001A": credit_amount,
            "syn_annuity_1002A": np.round(credit_amount * debt_ratio / 12 + 100, 2),
            "syn_applicationcnt_1003L": rng.poisson(1.5, size=n).astype(np.int64),
            "syn_clienttype_1004M": rng.choice(_CLIENT_TYPE_HASHES, size=n, p=(0.6, 0.3, 0.1)),
        }
    )
    for stem in _write_chunks(static0, train_dir, "static", 0, 2):
        record(stem, pl.read_parquet(train_dir / f"{stem}.parquet"))
    definitions |= {
        "syn_credamount_1001A": "[synthetic] Credit amount applied for.",
        "syn_annuity_1002A": "[synthetic] Monthly instalment of the applied loan.",
        "syn_applicationcnt_1003L": "[synthetic] Number of applications in the last year.",
        "syn_clienttype_1004M": "[synthetic] Client type (masked).",
    }

    # ---- static_cb_0: depth 0, one file, one near-empty column ----
    risk_missing = rng.random(n) < spec.riskassessment_missing_share
    static_cb = pl.DataFrame(
        {
            "case_id": case_id,
            "riskassesment_940T": pl.Series(
                np.where(risk_missing, np.nan, np.round(rng.normal(0.2, 1.0, size=n), 3))
            ).fill_nan(None),
            "syn_cbinquiries_1010L": rng.poisson(2.0, size=n).astype(np.int64),
            "syn_cbdebtoutstand_1011A": np.round(income * debt_ratio * rng.uniform(0.5, 2.0, size=n), 2),
        }
    )
    static_cb.write_parquet(train_dir / "train_static_cb_0.parquet")
    record("train_static_cb_0", static_cb)
    definitions |= {
        "riskassesment_940T": "Normalized risk assessed by the credit bureau.",
        "syn_cbinquiries_1010L": "[synthetic] Credit bureau inquiries in the last year.",
        "syn_cbdebtoutstand_1011A": "[synthetic] Outstanding debt reported by the bureau.",
    }

    # ---- person_1: depth 1; num_group1 == 0 is the applicant, >0 related persons ----
    n_related = rng.integers(0, 3, size=n)
    p_case, p_group = [], []
    for cid, k in zip(case_id, n_related):
        for g in range(k + 1):
            p_case.append(cid)
            p_group.append(g)
    p_case_arr = np.array(p_case, dtype=np.int64)
    p_group_arr = np.array(p_group, dtype=np.int64)
    is_applicant = p_group_arr == 0
    idx_of_case = {cid: i for i, cid in enumerate(case_id)}
    owner = np.array([idx_of_case[c] for c in p_case_arr])
    m = len(p_case_arr)

    def applicant_only(values: np.ndarray) -> list:
        return [v if a else None for v, a in zip(values.tolist(), is_applicant)]

    # The real birth dates are truncated to the first of the month.
    birth_exact = [decision[i] - timedelta(days=int(age_years[i] * 365.25)) for i in owner]
    birth = [_iso(d.replace(day=1)) for d in birth_exact]
    person = pl.DataFrame(
        {
            "case_id": p_case_arr,
            "num_group1": p_group_arr,
            "birth_259D": applicant_only(np.array(birth, dtype=object)),
            "education_927M": applicant_only(rng.choice(_EDUCATION_HASHES, size=m, p=_EDUCATION_P)),
            "empl_employedtotal_800L": applicant_only(rng.choice(_EMPLOYED_TOTAL, size=m, p=_EMPLOYED_TOTAL_P)),
            "empl_industry_691L": applicant_only(rng.choice(_INDUSTRY, size=m, p=_INDUSTRY_P)),
            "familystate_447L": applicant_only(rng.choice(_FAMILY, size=m, p=_FAMILY_P)),
            "incometype_1044T": applicant_only(rng.choice(_INCOME_TYPE, size=m, p=_INCOME_TYPE_P)),
            "mainoccupationinc_384A": applicant_only(income[owner]),
            "sex_738L": applicant_only(rng.choice(_SEX, size=m, p=_SEX_P)),
        },
        schema_overrides={"mainoccupationinc_384A": pl.Float64},
    )
    person.write_parquet(train_dir / "train_person_1.parquet")
    record("train_person_1", person)
    definitions |= {
        "birth_259D": "Birth date of the client.",
        "education_927M": "Education level of the client (masked).",
        "empl_employedtotal_800L": "Employment length.",
        "empl_industry_691L": "Employment industry.",
        "familystate_447L": "Family state of the client.",
        "incometype_1044T": "Type of income of the client.",
        "mainoccupationinc_384A": "Income of the client's main occupation.",
        "sex_738L": "Gender of the client.",
    }

    # ---- applprev_1: depth 1, two row chunks, planted leakage ----
    n_prev = rng.poisson(2.0, size=n)
    a_rows = []
    for i, (cid, k) in enumerate(zip(case_id, n_prev)):
        for g in range(k):
            a_rows.append((i, cid, g))
    a_owner = np.array([r[0] for r in a_rows], dtype=np.int64)
    a_case = np.array([r[1] for r in a_rows], dtype=np.int64)
    a_group = np.array([r[2] for r in a_rows], dtype=np.int64)
    na = len(a_rows)
    leak = rng.random(na) < spec.leak_share
    offsets = np.where(leak, rng.integers(1, 200, size=na), -rng.integers(10, 1500, size=na))
    approval = [_iso(decision[o] + timedelta(days=int(off))) for o, off in zip(a_owner, offsets)]
    # Created up to 30 days before approval, but never on or before the decision
    # for a leaked record, so the anchor rule drops exactly the leaked rows.
    gaps = np.where(leak, np.minimum(rng.integers(0, 31, size=na), offsets - 1), rng.integers(0, 31, size=na))
    creation = [_iso(decision[o] + timedelta(days=int(off - g))) for o, off, g in zip(a_owner, offsets, gaps)]
    prev_dpd = np.where(has_dpd[a_owner] & (rng.random(na) < 0.6), worst_dpd[a_owner], 0)
    applprev = pl.DataFrame(
        {
            "case_id": a_case,
            "num_group1": a_group,
            "actualdpd_943P": prev_dpd.astype(np.float64),
            "annuity_853A": np.round(income[a_owner] * rng.uniform(0.02, 0.2, size=na), 2),
            "approvaldate_319D": approval,
            "creationdate_885D": creation,
            "cancelreason_3545846M": rng.choice(_CANCEL_HASHES, size=na, p=(0.7, 0.2, 0.1)),
            "byoccupationinc_3656910L": np.round(income[a_owner] * rng.uniform(0.8, 1.2, size=na), -2),
        }
    )
    for stem in _write_chunks(applprev, train_dir, "applprev", 1, 2):
        record(stem, pl.read_parquet(train_dir / f"{stem}.parquet"))
    definitions |= {
        "actualdpd_943P": "Days past due of the previous contract (actual).",
        "annuity_853A": "Monthly annuity for previous applications.",
        "approvaldate_319D": "Approval date of the previous application.",
        "creationdate_885D": "Date when the previous application was created.",
        "cancelreason_3545846M": "Cancellation reason of the previous application (masked).",
        "byoccupationinc_3656910L": "Applicant's income from previous applications.",
    }

    # ---- credit_bureau_a_2: depth 2 (contract x month), two row chunks ----
    b_rows = []
    for i, cid in enumerate(case_id):
        for contract in range(int(rng.integers(0, 4))):
            for month in range(int(rng.integers(1, 7))):
                b_rows.append((i, cid, contract, month))
    b_owner = np.array([r[0] for r in b_rows], dtype=np.int64)
    nb = len(b_rows)
    b_month = np.array([r[3] for r in b_rows], dtype=np.int64)
    # np.where evaluates both branches, so the upper bound must be valid for every row.
    dpd_draw = rng.integers(1, np.maximum(worst_dpd[b_owner], 1) + 1)
    b_dpd = np.where(has_dpd[b_owner] & (rng.random(nb) < 0.4), dpd_draw, 0)
    bureau2 = pl.DataFrame(
        {
            "case_id": np.array([r[1] for r in b_rows], dtype=np.int64),
            "num_group1": np.array([r[2] for r in b_rows], dtype=np.int64),
            "num_group2": b_month,
            "syn_pmtsdpd_1031P": b_dpd.astype(np.float64),
            "syn_pmtsoverdue_1032A": np.round(np.where(b_dpd > 0, rng.uniform(100, 5000, size=nb), 0.0), 2),
            "syn_pmtsdate_1033D": [
                _iso(decision[o] - timedelta(days=int(30 * (mo + 1)))) for o, mo in zip(b_owner, b_month)
            ],
        }
    )
    for stem in _write_chunks(bureau2, train_dir, "credit_bureau_a", 2, 2):
        record(stem, pl.read_parquet(train_dir / f"{stem}.parquet"))
    definitions |= {
        "syn_pmtsdpd_1031P": "[synthetic] Days past due for a monthly payment of a bureau contract.",
        "syn_pmtsoverdue_1032A": "[synthetic] Overdue amount for a monthly payment of a bureau contract.",
        "syn_pmtsdate_1033D": "[synthetic] Date of the monthly payment record.",
    }

    # ---- tax registries: one source, three successive providers (C4) ----
    # c covers early decisions, a the middle period, b the late period, with
    # short overlaps; real column names; a few records dated after the decision.
    periods = {
        "tax_registry_c_1": (date(2018, 1, 1), date(2019, 9, 30)),
        "tax_registry_a_1": (date(2019, 8, 15), date(2020, 4, 15)),
        "tax_registry_b_1": (date(2020, 4, 1), date(2030, 1, 1)),
    }
    names = {
        "tax_registry_a_1": ("amount_4527230A", "recorddate_4527225D", "name_4527232M"),
        "tax_registry_b_1": ("amount_4917619A", "deductiondate_4917603D", "name_4917606M"),
        "tax_registry_c_1": ("pmtamount_36A", "processingdate_168D", "employername_160M"),
    }
    tax_leak_rows = 0
    for stem, (lo, hi) in periods.items():
        amount_col, date_col, employer_col = names[stem]
        rows = [(i, cid, g) for i, cid in enumerate(case_id) if lo <= decision[i] < hi
                for g in range(int(rng.integers(0, 4)))]
        owner = np.array([r[0] for r in rows], dtype=np.int64)
        nt = len(rows)
        late = rng.random(nt) < 0.03
        tax_leak_rows += int(late.sum())
        offsets_t = np.where(late, -rng.integers(1, 15, size=nt), rng.integers(30, 400, size=nt))
        tax = pl.DataFrame(
            {
                "case_id": np.array([r[1] for r in rows], dtype=np.int64),
                "num_group1": np.array([r[2] for r in rows], dtype=np.int64),
                amount_col: np.round(income[owner] / 12 * rng.uniform(0.1, 0.3, size=nt), 2),
                date_col: [_iso(decision[o] - timedelta(days=int(d))) for o, d in zip(owner, offsets_t)],
                employer_col: rng.choice(_EMPLOYER_HASHES, size=nt),
            },
            schema_overrides={amount_col: pl.Float64, date_col: pl.String, employer_col: pl.String},
        )
        tax.write_parquet(train_dir / f"train_{stem}.parquet")
        record(f"train_{stem}", tax)
    definitions |= {
        "amount_4527230A": "Tax deductions amount tracked by the government registry.",
        "recorddate_4527225D": "Date of tax deduction record.",
        "name_4527232M": "Name of employer.",
        "amount_4917619A": "Tax deductions amount tracked by the government registry.",
        "deductiondate_4917603D": "Tax deduction date.",
        "name_4917606M": "Name of employer.",
        "pmtamount_36A": "Tax deductions amount for credit bureau payments.",
        "processingdate_168D": "Date when the tax deduction is processed.",
        "employername_160M": "Employer's name.",
    }

    # ---- feature definitions and manifest ----
    pl.DataFrame(
        {"Variable": list(definitions), "Description": list(definitions.values())}
    ).write_csv(root / "feature_definitions.csv")

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "spec": asdict(spec),
        "tables": dict(sorted(tables.items())),
        "n_cases": n,
        "observed_default_rate": float(target.mean()),
        "leakage": {
            "table_family": "applprev",
            "column": "approvaldate_319D",
            "n_rows_after_decision": int(leak.sum()),
        },
        "tax_rows_after_decision": tax_leak_rows,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return SyntheticDataset(root=root, train_dir=train_dir, manifest=manifest)
