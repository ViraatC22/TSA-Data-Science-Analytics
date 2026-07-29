# Project Audit

## 1. Project purpose

This repository supports "The Global Standstill," a TSA Data Science and
Analytics project about changes in international tourism during the COVID-19
pandemic. Its coherent software scope is an interactive Streamlit dashboard plus
reusable, testable calculations over a public country-level tourism dataset.

The supplied analysis is descriptive. It does not establish that transport mode
or policy caused the measured regional differences, and it does not contain a
trained forecasting model.

## 2. Evidence reviewed

- `README.md` and the complete Streamlit application.
- Both existing Git commits and the public GitHub repository.
- `Inbound Tourism-Transport.csv`, including schema, missingness, series keys,
  target-country observations, file digest, and source-license research.
- The complete 23-page local portfolio PDF, including visual inspection of its
  methodology, charts, findings, and bibliography.
- The existing virtual environment and documented setup/deployment statements.

## 3. Existing architecture

- `TSA_Data_Science_Analytics.py` is a 552-line monolithic Streamlit application.
  It performs ingestion, reshaping, regional classification, calculations, chart
  creation, filtering, KPI calculation, export, and interface rendering.
- `Inbound Tourism-Transport.csv` is the only runtime data source.
- `README.md` describes installation and three headline findings.
- There is no analysis module, dependency manifest, test suite, verification
  command, or CI workflow.

## 4. Dataset provenance and privacy

The tracked CSV is a renamed copy of the Kaggle dataset "UNWTO Tourism Data -
Structured for Analysis":

- Upstream URL:
  `https://www.kaggle.com/datasets/tronheim/unwto-tourism-data-structured-for-analysis`
- Upstream author: `tronheim` / Amin
- Upstream description: an aggregation derived from UN Tourism (UNWTO) data
- Upstream license shown by Kaggle: Open Database License for the database and
  Database Contents License for its contents
- Local SHA-256:
  `7b8d8b684b662269737c40380d11886069daf461b0b202eafdd241d6c2420a98`
- Local shape: 8,253 rows and 33 columns
- Coverage: 223 countries/territories and annual columns from 1995 through 2022

The file contains aggregate country statistics, not person-level records. No
direct personal information, credentials, tokens, private keys, or local
databases were found. The dataset was already committed and published in the
repository's first commit.

The structured file omits the original `Units` and source-note fields. The
dashboard must therefore state its unit assumptions explicitly and must not
combine unrelated series.

## 5. Local source PDF handling

`TSA DATA SCIENCE INTERSCHOOL.pdf` is a local, untracked 23-page competition
portfolio with team names and submission material.

- SHA-256:
  `dba96a72fc2bbb9d68933291bf7faf180aab9c9630ccb708bc6f9ba087d6a9a6`
- Size: 605,151 bytes
- Runtime requirement: none
- Decision: retain it unchanged on the local filesystem and explicitly ignore it
  in Git. Do not publish it automatically.

The PDF is treated as a historical requirements and claim source. Corrections
belong in code and documentation; the submitted artifact is not silently
rewritten.

## 6. Reproducibility and correctness findings

### 6.1 Overlapping expenditure series are double-counted

The main regional charts select both `Travel` and `Total` rows from
`Inbound Tourism-Expenditure` and sum them. `Total` is an aggregate series while
`Travel` is one of its components, so this combination double-counts receipts.
The code's computed values do not reproduce several headline values in the
README or portfolio.

The corrected dashboard will use the `Travel` line item consistently because it
has broader coverage across the selected countries and is directly identifiable
in the source file. It will label that scope as travel receipts rather than total
tourism expenditure.

### 6.2 Published headline values do not match the current data/code

Using the repository's region map and the non-overlapping `Travel` series:

| Region | 2019 receipts | 2020 change | 2022 recovery vs. 2019 |
| --- | ---: | ---: | ---: |
| Asia | 209,155 USD millions | -74.5% | 32.4% |
| Europe | 356,428 USD millions | -58.7% | 93.5% |
| North America | 253,378 USD millions | -61.7% | 74.6% |

These are descriptive aggregates over the countries present in the hard-coded
three-bloc sample; they are not official regional totals.

### 6.3 Units and series identity are obscured

- Expenditure values are USD millions.
- Transport and accommodation counts in the upstream data are reported in
  thousands.
- The current explorer can combine rows from multiple report types,
  subcategories, and units under one metric label.
- A chart labeled as a count can therefore display values measured in thousands,
  and a line can contain multiple observations for one country/year.

The repaired explorer must require one report/category/subcategory/metric series
and show the inferred unit.

### 6.4 One valid number is silently discarded

The Italy 2022 domestic same-day visitor value is stored as `46 264` with a
non-breaking-space thousands separator. The current numeric coercion turns it
into missing data. Ingestion must normalize supported thousands separators and
reject unexpected non-numeric tokens.

### 6.5 Runtime and error-handling gaps

- Data loading depends on the process working directory instead of the script's
  location.
- Only missing-file errors are handled; schema and duplicate-series defects are
  not surfaced.
- Percentage changes can divide by zero.
- Missing base years can produce empty or misleading charts.
- `add_regions` mutates its caller's DataFrame.
- The checked-in README references a nonexistent `requirements.txt`.
- The existing `.venv` points to a removed Python 3.12 interpreter and is not
  reproducible.

## 7. Methodology gaps

- No unit tests exist despite the portfolio stating that calculations were
  unit-tested.
- The regional sample is a hand-selected three-bloc taxonomy, not a complete
  UN Tourism regional classification.
- Coverage varies by country, series, and year; raw sums are not adjusted to a
  common balanced country panel.
- Current-dollar expenditure is not adjusted for inflation or exchange rates.
- The dashboard supports descriptive comparisons only. The PDF's causal and
  forecast-validation language is stronger than the supplied analysis supports.
- The dataset ends in 2022, so claims about 2023-2025 are external commentary,
  not outputs validated by this repository.

## 8. Security and accessibility

- No secrets, authentication flow, network calls, or user uploads exist.
- CSV export exposes only the same aggregate data already present in the public
  repository.
- Hard-coded HTML is passed to Streamlit, but no user-controlled value is
  interpolated into that HTML.
- Charts rely substantially on color and require more explicit units, scopes,
  and empty-state handling for usability.

## 9. Deployment status

The PDF references a Streamlit Community Cloud URL. The existing repository is
public, its default branch is `main`, and the deployment depends on that
repository. Changing visibility could disrupt the published app and is not
required for aggregate-data privacy, so recovery will preserve the existing
remote visibility and document it rather than silently changing it.

## 10. Completion definition

The project is complete when:

1. A strict, reusable analysis module loads the local dataset relative to the
   repository and preserves valid formatted numbers.
2. Calculations use one clearly identified source series with explicit units.
3. Regional shock/recovery and country mechanism figures are reproducible from
   tested functions.
4. The explorer prevents incompatible-series aggregation.
5. Empty, missing-base, zero-denominator, schema, and duplicate cases are safe.
6. Dependencies are declared and one canonical command runs compile, tests, and
   a headless Streamlit startup smoke.
7. The principal dashboard flow is verified in a browser.
8. README, provenance, limitations, source-PDF handling, Git, remote, CI, and
   final status match reality.

## 11. Implementation plan and assumptions

- Preserve the original visual concept and Streamlit stack.
- Extract ingestion and statistical logic into `tsa_analysis.py`.
- Use `Inbound Tourism-Expenditure` / `Travel` for regional receipt figures.
- Keep the existing transparent country-to-region map, but label it as a sample.
- Use exact series keys in the explorer and expose coverage/units.
- Add deterministic unit tests plus regression checks against the tracked CSV.
- Keep the authored PDF local-only and keep the already published aggregate CSV.
- Correct unsupported headline claims in the README and dashboard rather than
  modifying the historical submission PDF.
