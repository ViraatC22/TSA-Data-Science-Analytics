# The Global Standstill

Interactive, reproducible analysis of changes in international tourism during the
COVID-19 pandemic. The project was created for the TSA Data Science and
Analytics competition by Viraat Chauhan and Pranav Sreepada.

**Status:** recovered and locally verified. The dashboard now separates
validated statistical calculations from Streamlit rendering, uses one
non-overlapping expenditure series, exposes units and sample coverage, and
prevents incompatible source series from being combined.

This is a descriptive educational analysis. It does not establish that transport
mode or public policy caused the observed differences, and it does not contain a
trained forecast model.

## What the dashboard provides

- Ten portfolio charts covering receipt trends, the 2020 shock, recovery,
  transport modes, accommodation, cross-border financial flows, and country
  comparisons.
- A validated data explorer that requires an exact
  report/category/subcategory/metric series before calculation.
- Explicit unit, country count, observation count, and selected-series context.
- Safe KPIs for cumulative value, peak aggregate year, and latest comparable
  change.
- Download of only the exact filtered series shown in the explorer.
- Clear empty states for missing baselines or observations.

## Corrected analytical scope

The main regional figures use only:

```text
Report Type: Inbound Tourism-Expenditure
Metric: Travel
Unit: current USD millions
```

The original dashboard summed both `Travel` and `Total`. Those are overlapping
component/aggregate series, so combining them double-counted receipts.

The three regions are transparent selected-market groups, not complete official
UN Tourism regional totals. Countries with a 2019 `Travel` observation are:

- North America: Canada, Mexico, United States of America
- Europe: Austria, France, Germany, Greece, Italy, Portugal, Spain, United Kingdom
- Asia: China, India, Indonesia, Japan, Malaysia, Thailand

Using that scope, the reproducible headline values are:

| Region | 2019 receipts | 2020 change | 2022 level vs. 2019 |
| --- | ---: | ---: | ---: |
| Asia | 209,155 USD millions | -74.5% | 32.4% |
| Europe | 356,428 USD millions | -58.7% | 93.5% |
| North America | 253,378 USD millions | -61.7% | 74.6% |

These values are regression-tested against the tracked CSV.

## Data source, license, and privacy

The runtime file `Inbound Tourism-Transport.csv` is a renamed copy of
[UNWTO Tourism Data - Structured for Analysis on Kaggle](https://www.kaggle.com/datasets/tronheim/unwto-tourism-data-structured-for-analysis),
an aggregation derived from UN Tourism data.

- Upstream author: `tronheim` / Amin
- Upstream license: Open Database License (database) and Database Contents
  License (contents), as displayed by Kaggle
- Local SHA-256:
  `7b8d8b684b662269737c40380d11886069daf461b0b202eafdd241d6c2420a98`
- Shape: 8,253 statistical series and 33 columns
- Coverage: 223 countries/territories; annual columns from 1995 through 2022
- Long-form non-missing observations after validated parsing: 91,514

The file contains aggregate country statistics, not person-level records. It has
no direct personal information or credentials. The structured source omits its
original `Units` and source-note columns, so the application states exact units
only where supported by the upstream description:

- expenditure: USD millions;
- inbound transport: thousands of arrivals;
- accommodation guests/overnights: thousands; and
- several other series: explicitly labeled as source units not included in the
  structured file.

The local competition portfolio `TSA DATA SCIENCE INTERSCHOOL.pdf` is preserved
unchanged and intentionally ignored by Git. It contains team-identifying
submission material, is not required by the application, and is treated as a
historical source artifact rather than a runtime dependency.

See [`docs/PROJECT_AUDIT.md`](docs/PROJECT_AUDIT.md) for the full provenance,
privacy, and methodology review.

## Requirements

- Python 3.11 or newer recommended
- Dependencies listed in `requirements.txt`

## Installation

```bash
git clone https://github.com/ViraatC22/TSA-Data-Science-Analytics.git
cd TSA-Data-Science-Analytics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The previous local `.venv` referenced a removed Python interpreter. Recreate it
with the commands above rather than relying on that stale environment.

## Run locally

```bash
python -m streamlit run TSA_Data_Science_Analytics.py
```

Streamlit prints the local URL, normally `http://localhost:8501`.

The portfolio's historical deployment URL is
[tsa-data-science-analytics-interschool.streamlit.app](https://tsa-data-science-analytics-interschool.streamlit.app/).
As of 2026-07-29 it redirects to Streamlit authorization, so public access to
that hosted instance is not verified. Local operation does not require an
account or credentials.

## Verification

Run the canonical quality gate:

```bash
make verify
```

It performs:

1. Python bytecode compilation;
2. ten deterministic unit, real-data regression, chart, and Streamlit flow
   tests; and
3. a real headless Streamlit server health check.

The tests cover schema failures, duplicate series, formatted thousands,
non-numeric tokens, zero denominators, ambiguous explorer selections, corrected
regional values, transport/accommodation/financial series, all ten Vega chart
specifications, and both dashboard modules.

## Architecture

```text
TSA_Data_Science_Analytics.py  Streamlit presentation and interaction
tsa_analysis.py                Validated loader and reusable statistics
Inbound Tourism-Transport.csv Tracked aggregate source data
tests/                         Unit, regression, chart, and app-flow tests
scripts/smoke_streamlit.py     Real server startup check
.github/workflows/verify.yml   Python verification matrix
docs/                          Audit, completion plan, and final handoff
```

The data path is resolved relative to the application file, so the dashboard can
be launched from a different working directory.

## Methodology limitations

- The country groups are a hand-selected competition taxonomy, not exhaustive
  official regions.
- Country and year availability varies. Aggregate sums use available
  observations and are not forced into a balanced sample.
- Expenditure is in current USD and is not adjusted for inflation or exchange
  rates.
- The source ends in 2022. Statements about 2023-2025 are not validated by the
  supplied data or code.
- No causal design controls for policy, income, disease burden, geography,
  exchange rates, or other confounders.
- No trained forecasting model, uncertainty interval, or out-of-sample forecast
  evaluation is included.
- The original competition PDF contains historical claims that differ from the
  corrected reproducible dashboard; it has been preserved rather than silently
  rewritten.

## Security and licensing

The app makes no external API calls, accepts no uploads, and uses no secrets.
CSV download contains only the aggregate observations already present in the
repository.

No license has been declared for the project code, so all code rights remain
with the authors. The dataset retains its upstream database/content licensing
terms and attribution requirements.

The GitHub repository predates this recovery and is public because the existing
Streamlit deployment depends on it. Visibility was preserved rather than changed
silently. The local source PDF remains excluded.
