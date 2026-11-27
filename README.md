# Faithful Credit Reasons

When a lender rejects an application, it must tell the applicant the main
reasons, and those reasons must accurately describe what its model actually
scored. Banks now want LLMs to write those letters. This project builds an
out-of-time-validated credit default model on Home Credit's 2024 stability
benchmark, adds an LLM reason-writing layer, and **uses the model itself as
ground truth** to measure whether LLM-written explanations are faithful. It
then shows which design choices, in the model and in the generation pipeline,
make them so.

> Status: under construction (C1 of 28). Numbers appear here only once they
> are measured. The full specification is in [REFERENCE.md](REFERENCE.md).

## Thesis

Code decides what the reasons are, the LLM only puts them into words, and a
deterministic verifier checks every claim against the model (re-scoring it for
any recourse advice) before a letter is released.

The project tests three claims, each with a decision rule fixed in advance:

1. **Ungrounded LLM explanations of credit decisions are fluent but measurably
   unfaithful**, and an LLM grader under-detects their errors relative to
   model-in-the-loop verification.
2. **Structured grounding plus deterministic verification makes explanations
   faithful** at a small readability cost.
3. **Explanation quality is set upstream in the ML design**: monotone
   constraints and the SHAP reference population change reason stability and
   recourse validity, at a measurable AUC cost.

Any of the three can fail, and a failed claim is reported as a result.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/fcr/` | The installable package: data, features, models, monitoring, explain, generate, verify, llm, eval |
| `configs/config.yaml` | Every tunable value, validated on load |
| `scripts/smoke_cNN.py` | One smoke run per commit, against real data or APIs where relevant |
| `tests/` | Runs on synthetic, competition-shaped fixtures; no data download needed |
| `notebooks/` | Thin Kaggle drivers that install a pinned commit and run a script |
| `results/` | Metrics, tables and figures (small files only) |

## Local setup

Requires Python 3.12. On macOS, LightGBM also needs OpenMP: `brew install libomp`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # exact, tested versions
pip install -e . --no-deps          # the fcr package itself
python -m pytest
python scripts/smoke_c01.py
```

Heavy data work runs on Kaggle; see [notebooks/README.md](notebooks/README.md).
The competition data is not redistributed here.

## License

MIT
