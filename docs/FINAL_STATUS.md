# Final Project Status

## Final status

**COMPLETED AS FAR AS OBJECTIVELY POSSIBLE**

The local analysis and Streamlit application are reproducible, tested,
documented, version-controlled, synchronized to GitHub, and protected by CI. The
historical hosted Streamlit URL currently requires authorization; making that
deployment publicly reachable requires action from its Streamlit owner.

## Original condition

- The dashboard was a monolithic 552-line script with ingestion, calculations,
  charts, and interface logic intertwined.
- Regional expenditure charts summed overlapping `Travel` and `Total` rows,
  double-counting receipts.
- Several headline values and causal/forecast claims were not reproducible from
  the supplied code and 1995-2022 dataset.
- The explorer could combine report types, subcategories, and incompatible units.
- Units were missing or misleading for transport and accommodation counts.
- A valid number containing a non-breaking-space thousands separator was silently
  discarded.
- Data loading depended on the process working directory.
- The README referenced a nonexistent dependency file.
- The existing local virtual environment pointed to a removed interpreter.
- No tests, canonical verification command, or CI existed.

## Completed work

- Preserved the existing Git history, valid remote, tracked dataset, and local
  competition PDF before changes.
- Audited the 23-page PDF, dataset provenance/license, privacy, secrets, file
  sizes, code, public remote, deployment, and historical claims.
- Added a strict long-form loader with schema, identifier, duplicate-series, and
  numeric-token validation.
- Preserved supported comma/non-breaking-space thousands separators.
- Made the dataset path location-independent.
- Extracted reusable, Streamlit-independent statistics into `tsa_analysis.py`.
- Replaced overlapping aggregation with the exact inbound `Travel` receipt line
  item and labeled the selected-country sample.
- Added safe grouped changes, recovery indices, country comparisons, transport,
  accommodation, financial-flow, unit, and KPI calculations.
- Rebuilt all ten charts on the validated functions with explicit units, scope,
  coverage, tooltips, and empty states.
- Rebuilt the explorer to require an exact
  report/category/subcategory/metric series.
- Replaced unsupported causal and forecast framing in the app/README with honest
  descriptive scope and limitations.
- Added a dependency manifest, canonical quality gate, real server smoke, CI,
  audit, completion plan, and operational README.

## Corrected reproducible findings

Using selected countries with a 2019 inbound `Travel` observation:

| Region | 2019 receipts | 2020 change | 2022 level vs. 2019 |
| --- | ---: | ---: | ---: |
| Asia | 209,155 USD millions | -74.5% | 32.4% |
| Europe | 356,428 USD millions | -58.7% | 93.5% |
| North America | 253,378 USD millions | -61.7% | 74.6% |

These are descriptive selected-market sums, not official complete regional
totals or estimates of causal policy effects.

## Tests and verification

The suite contains ten deterministic tests covering:

- required schema, duplicate-series, malformed-token, and ambiguous-selection
  rejection;
- comma and non-breaking-space thousands parsing;
- real-data shape, year range, and the recovered Italy observation;
- regional golden values, changes, and recovery indices;
- transport, accommodation, and financial-flow series;
- zero/non-finite percentage baselines and empty KPI behavior;
- compilation of all ten Altair/Vega chart specifications; and
- both Streamlit portfolio and explorer flows.

Verified on 2026-07-29:

| Check | Result |
| --- | --- |
| Fresh Python 3.14 virtual environment | Dependency install passed |
| `make verify` | Compilation, 10 tests, and server smoke passed |
| Streamlit `AppTest` portfolio flow | Passed |
| Streamlit `AppTest` explorer flow | Passed |
| Real headless server health endpoint | Passed |
| GitHub CI Python 3.11 | Passed |
| GitHub CI Python 3.13 | Passed |

CI evidence:
`https://github.com/ViraatC22/TSA-Data-Science-Analytics/actions/runs/30476295911`

The in-app browser surface was unavailable during recovery, so a manual visual
click-through could not be recorded. Native Streamlit AppTest covered both
interactive modules, and the real HTTP server health check passed.

## Dataset and source handling

### Tracked aggregate CSV

- File: `Inbound Tourism-Transport.csv`
- Git state: already tracked and published in the first repository commit
- Privacy: aggregate country-level statistics; no person-level records
- SHA-256:
  `7b8d8b684b662269737c40380d11886069daf461b0b202eafdd241d6c2420a98`
- Recovery action: retained byte-for-byte unchanged
- Source:
  `https://www.kaggle.com/datasets/tronheim/unwto-tourism-data-structured-for-analysis`
- Upstream license shown by Kaggle: Open Database License / Database Contents
  License

### Local competition PDF

- File: `TSA DATA SCIENCE INTERSCHOOL.pdf`
- Git state: local-only and ignored
- SHA-256:
  `dba96a72fc2bbb9d68933291bf7faf180aab9c9630ccb708bc6f9ba087d6a9a6`
- Recovery action: visually inspected and retained byte-for-byte unchanged
- Rationale: historical authored submission with team-identifying material; not a
  runtime dependency and not automatically published

No credentials, private keys, direct personal dataset records, or production
secrets were found or committed.

## Documentation

- `README.md`: corrected scope/findings, data source/license/privacy, setup,
  operation, verification, architecture, deployment, and limitations.
- `docs/PROJECT_AUDIT.md`: forensic code/data/PDF review and completion
  definition.
- `docs/COMPLETION_PLAN.md`: task-level recovery record and deferred scope.
- `docs/FINAL_STATUS.md`: this handoff and verification report.

## GitHub and branch status

- Repository: `https://github.com/ViraatC22/TSA-Data-Science-Analytics`
- Visibility: **PUBLIC** (pre-existing and preserved)
- Default/final branch: `main`
- Verified implementation/documentation commit:
  `c7cf77a5b61696ee97e09c4da39db7b923881f94`
- At verification time, local `main` and `origin/main` matched with a clean
  working tree.

The repository was not changed to private because its public visibility predates
the recovery, the data is aggregate and upstream-licensed, and the existing
Streamlit deployment depends on the repository. The local PDF remains excluded.

## Deployment status

The project is deployment-ready as a local Streamlit application.

Historical URL:
`https://tsa-data-science-analytics-interschool.streamlit.app/`

On 2026-07-29 the URL returned an HTTP redirect to Streamlit authorization.
Public hosted access is therefore **externally blocked**, while local operation
is fully usable.

Exact owner action:

1. Sign in to the Streamlit workspace that owns the app.
2. Confirm its GitHub repository access and main file path.
3. Set the app's sharing policy to the intended audience or redeploy it.
4. Run both portfolio and explorer flows in the hosted environment.

## Known limitations

- Region groups are a hand-selected competition taxonomy, not official complete
  regions.
- Available-country coverage varies by series and year.
- Current-USD expenditure is not inflation or exchange-rate adjusted.
- The dataset ends in 2022.
- No identified causal model or trained forecasting model exists.
- Source `Units` and notes were omitted from the structured CSV, so unsupported
  series are conservatively labeled.
- The historical PDF contains claims that differ from the corrected dashboard.
- The project code has no declared software license.

## Remaining blockers

No blocker remains for installation, local analysis, testing, or continued
development.

The only external blocker is hosted Streamlit access, which requires the
deployment owner's account and sharing decision.

## Recommended future enhancements

1. Have the deployment owner restore or intentionally retire the hosted app.
2. Add a small screenshot set after manual visual review.
3. Restore source-level unit/note metadata from a licensed upstream export.
4. Add balanced-panel and inflation-adjusted analysis options.
5. Add identified policy variables before making causal comparisons.
6. Add a newer dataset and a real forecast protocol before evaluating post-2022
   claims.
7. Choose and add a code license if redistribution is intended.
