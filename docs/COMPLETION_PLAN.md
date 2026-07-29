# Completion Plan

Statuses: `NOT_STARTED`, `IN_PROGRESS`, `COMPLETED`, `BLOCKED`,
`DEFERRED_WITH_REASON`.

## Milestone 0 - Preservation and evidence

| ID | Task | Reason | Acceptance criteria | Verification | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- |
| M0-1 | Reconcile Git, remote, source PDF, and data history | Preserve pre-recovery work and published dependencies | Existing commits/remote retained; recovery branch created | `git log`; `git status`; `gh repo view` | COMPLETED | Existing `5d8dd64` |
| M0-2 | Audit secrets, PII, size, provenance, and licenses | Avoid publishing private or inappropriately sourced material | Findings and hashes recorded; PDF ignored but untouched | File/content scans; Kaggle metadata; PDF inspection | COMPLETED | `7cba0cb` |

## Milestone 1 - Analysis correctness

| ID | Task | Reason | Acceptance criteria | Verification | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- |
| M1-1 | Add strict, location-independent data loading | Current loader loses a formatted value and depends on CWD | Required schema/unique series validated; supported separators parsed; exact path works | Unit and real-data tests | COMPLETED | `25e79d1` |
| M1-2 | Add reusable statistical functions | Calculations are embedded in UI code and untested | Safe percentage change, regional receipts, shock, recovery, balance, transport, and KPI functions | Unit/regression tests | COMPLETED | `25e79d1` |
| M1-3 | Remove overlapping-series aggregation | Existing figures double-count `Travel` and `Total` | Regional figures use one documented `Travel` series | Golden-value tests | COMPLETED | `25e79d1` |

## Milestone 2 - Dashboard integrity

| ID | Task | Reason | Acceptance criteria | Verification | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- |
| M2-1 | Wire charts to validated calculations | Charts duplicate fragile pivot/math logic | Ten chart functions handle missing/zero bases and label units/scope | Chart construction tests; AppTest | COMPLETED | `25e79d1` |
| M2-2 | Make explorer series-safe | Current explorer mixes report types/subcategories/units | User selects one exact report/category/subcategory/metric series | AppTest for both modules | COMPLETED | `25e79d1` |
| M2-3 | Improve methodology and empty-state guidance | Existing UI overstates causal/forecast conclusions | Descriptive scope, coverage, units, and limitations are visible | AppTest; source review | COMPLETED | `25e79d1` |

## Milestone 3 - Reproducible quality gate

| ID | Task | Reason | Acceptance criteria | Verification | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- |
| M3-1 | Declare supported dependencies | README references a missing file; stale venv is unusable | Clean install succeeds from a dependency manifest | Fresh temporary Python 3.14 environment | COMPLETED | `25e79d1` |
| M3-2 | Add deterministic tests and canonical verification | Portfolio claims tests but none exist | One command compiles, runs meaningful tests, and smoke-starts Streamlit | `make verify` | COMPLETED | `25e79d1` |
| M3-3 | Add/repair CI | Protect public main from regression | CI performs the same core gate on supported Python | GitHub Actions | IN_PROGRESS | `8ff8a7d` |

## Milestone 4 - Documentation and handoff

| ID | Task | Reason | Acceptance criteria | Verification | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- |
| M4-1 | Rewrite README and document provenance | Current setup and findings are inaccurate | Commands, data source/license, corrected findings, scope, risks, and structure match code | Command/link review | IN_PROGRESS | Pending |
| M4-2 | Verify app, merge, push, and report | Finish the recovery without breaking main | App flow and CI pass; clean local/remote main; exact report created | AppTest; server smoke; Git/gh checks | IN_PROGRESS | Pending |

## Deferred scope

| ID | Task | Status | Reason |
| --- | --- | --- | --- |
| D-1 | Rewrite or republish competition PDF | DEFERRED_WITH_REASON | It is a historical authored submission artifact; recovery preserves it unchanged. |
| D-2 | Infer causality from descriptive aggregates | DEFERRED_WITH_REASON | Requires identified policy/transport variables, controls, and a defensible causal design. |
| D-3 | Validate 2023-2025 forecast claims | DEFERRED_WITH_REASON | The tracked dataset ends in 2022 and no forecasting model or later dataset is supplied. |
| D-4 | Rewrite Git history to remove the aggregate CSV | DEFERRED_WITH_REASON | The ODbL/DbCL-licensed aggregate data contains no PII and is already public; destructive rewriting is unwarranted. |
