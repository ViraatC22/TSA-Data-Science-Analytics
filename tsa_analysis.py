"""Validated data and statistics for the TSA tourism dashboard.

The source CSV is a structured aggregation of UN Tourism data. Its identifying
columns describe a statistical series and its year columns contain observations.
This module keeps analysis logic independent of Streamlit so calculations are
testable and reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "Inbound Tourism-Transport.csv"

ID_COLUMNS = [
    "Country",
    "Report Type",
    "Category",
    "Subcategory",
    "Metric",
]

REGION_ORDER = ["North America", "Europe", "Asia"]
REGION_MAP = {
    # North America
    "UNITED STATES": "North America",
    "USA": "North America",
    "UNITED STATES OF AMERICA": "North America",
    "CANADA": "North America",
    "MEXICO": "North America",
    # Europe: selected major markets used by the original project
    "FRANCE": "Europe",
    "SPAIN": "Europe",
    "ITALY": "Europe",
    "GERMANY": "Europe",
    "UNITED KINGDOM": "Europe",
    "UK": "Europe",
    "AUSTRIA": "Europe",
    "GREECE": "Europe",
    "PORTUGAL": "Europe",
    # Asia: selected major markets used by the original project
    "THAILAND": "Asia",
    "CHINA": "Asia",
    "JAPAN": "Asia",
    "REPUBLIC OF KOREA": "Asia",
    "SOUTH KOREA": "Asia",
    "INDIA": "Asia",
    "INDONESIA": "Asia",
    "MALAYSIA": "Asia",
    "VIET NAM": "Asia",
}


class DataValidationError(ValueError):
    """Raised when the source file cannot support reliable calculations."""


@dataclass(frozen=True)
class ExplorerKpis:
    cumulative_value: float
    peak_year: int | None
    peak_year_value: float | None
    latest_year: int | None
    previous_year: int | None
    latest_change: float | None


def infer_unit(report_type: str, metric: str) -> str:
    """Infer the documented source unit for a statistical series.

    The structured Kaggle file omitted its source ``Units`` column, so only
    units supported by the upstream data description are stated precisely.
    """
    if "Expenditure" in report_type:
        return "USD millions"
    if report_type == "Inbound Tourism-Transport":
        return "thousand arrivals"
    if report_type in {"Domestic Tourism-Trips", "Outbound Tourism-Departures"}:
        return "thousand visitors"
    if "Tourism-Accommodation" in report_type:
        if metric == "Guests":
            return "thousand guests"
        if metric == "Overnights":
            return "thousand overnight stays"
    if report_type == "Tourism Industries":
        if metric.startswith("Occupancy rate"):
            return "percent"
        if metric == "Average length of stay":
            return "nights"
        if metric == "Available capacity (bed-places per 1000 inhabitans)":
            return "bed-places per 1,000 inhabitants"
        if metric.startswith("Number of"):
            return "reported units"
    return "source units not included in structured file"


def _parse_values(values: pd.Series) -> pd.Series:
    original = values.copy()
    cleaned = (
        values.astype("string")
        .str.strip()
        .str.replace("\u00a0", "", regex=False)
        .str.replace("\u202f", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    invalid = original.notna() & numeric.isna()
    if invalid.any():
        examples = sorted({str(value) for value in original.loc[invalid].head(5)})
        raise DataValidationError(
            f"source contains unsupported non-numeric values: {examples}"
        )
    return numeric


def load_tourism_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the wide source CSV into a validated, sorted long-form DataFrame."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"tourism data file not found: {source}")

    wide = pd.read_csv(source, dtype=str)
    if wide.columns.duplicated().any():
        raise DataValidationError("source contains duplicate column names")

    missing_columns = [column for column in ID_COLUMNS if column not in wide]
    if missing_columns:
        raise DataValidationError(
            f"source is missing required columns: {missing_columns}"
        )

    year_columns = sorted(
        (column for column in wide if column.isdigit() and len(column) == 4),
        key=int,
    )
    if not year_columns:
        raise DataValidationError("source must contain four-digit year columns")
    unexpected_columns = sorted(set(wide) - set(ID_COLUMNS) - set(year_columns))
    if unexpected_columns:
        raise DataValidationError(
            f"source contains unexpected columns: {unexpected_columns}"
        )

    for column in ID_COLUMNS:
        if wide[column].isna().any():
            raise DataValidationError(f"source contains missing {column!r} values")
        wide[column] = wide[column].str.strip()
        if wide[column].eq("").any():
            raise DataValidationError(f"source contains empty {column!r} values")

    duplicate_mask = wide.duplicated(ID_COLUMNS, keep=False)
    if duplicate_mask.any():
        examples = wide.loc[duplicate_mask, ID_COLUMNS].head(3).to_dict("records")
        raise DataValidationError(
            f"source contains duplicate statistical series: {examples}"
        )

    long = wide.melt(
        id_vars=ID_COLUMNS,
        value_vars=year_columns,
        var_name="Year",
        value_name="Value",
    )
    long["Value"] = _parse_values(long["Value"])
    long = long.dropna(subset=["Value"]).copy()
    long["Year"] = long["Year"].astype(int)
    long["Country"] = long["Country"].str.upper()
    long["Region"] = long["Country"].map(REGION_MAP)
    long["Unit"] = [
        infer_unit(report_type, metric)
        for report_type, metric in zip(
            long["Report Type"],
            long["Metric"],
            strict=True,
        )
    ]
    return long.sort_values([*ID_COLUMNS, "Year"], ignore_index=True)


def select_series(
    data: pd.DataFrame,
    *,
    report_type: str,
    metric: str,
    category: str | None = None,
    subcategory: str | None = None,
    countries: Iterable[str] | None = None,
    regions: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
    unique_by_country_year: bool = False,
) -> pd.DataFrame:
    """Select an exact source series without mutating the input DataFrame."""
    selected = data.loc[
        (data["Report Type"] == report_type) & (data["Metric"] == metric)
    ].copy()
    if category is not None:
        selected = selected.loc[selected["Category"] == category]
    if subcategory is not None:
        selected = selected.loc[selected["Subcategory"] == subcategory]
    if countries is not None:
        selected = selected.loc[selected["Country"].isin(list(countries))]
    if regions is not None:
        selected = selected.loc[selected["Region"].isin(list(regions))]
    if years is not None:
        selected = selected.loc[selected["Year"].isin(list(years))]

    if unique_by_country_year and selected.duplicated(["Country", "Year"]).any():
        raise DataValidationError(
            "selection is ambiguous; choose one category and subcategory"
        )
    return selected.reset_index(drop=True)


def safe_percent_change(current: float, baseline: float) -> float | None:
    """Return decimal change, or ``None`` for invalid/zero baselines."""
    if not np.isfinite(current) or not np.isfinite(baseline) or baseline == 0:
        return None
    return float((current - baseline) / baseline)


def regional_travel_receipts(
    data: pd.DataFrame,
    *,
    start_year: int = 2015,
    end_year: int = 2022,
) -> pd.DataFrame:
    """Aggregate the non-overlapping inbound ``Travel`` receipt line item.

    These are sums over the hand-selected countries in ``REGION_MAP``. They are
    not official complete regional totals.
    """
    receipts = select_series(
        data,
        report_type="Inbound Tourism-Expenditure",
        metric="Travel",
        regions=REGION_ORDER,
        years=range(start_year, end_year + 1),
        unique_by_country_year=True,
    )
    if receipts.empty:
        return pd.DataFrame(
            columns=["Region", "Year", "Value", "Country Count", "Unit"]
        )
    return (
        receipts.groupby(["Region", "Year"], as_index=False, observed=True)
        .agg(
            Value=("Value", "sum"),
            **{"Country Count": ("Country", "nunique")},
        )
        .assign(Unit="USD millions")
        .sort_values(["Region", "Year"], ignore_index=True)
    )


def grouped_change(
    values: pd.DataFrame,
    *,
    group_column: str,
    baseline_year: int,
    comparison_year: int,
    output_column: str = "Change",
) -> pd.DataFrame:
    """Calculate safe group changes between two years."""
    subset = values.loc[
        values["Year"].isin([baseline_year, comparison_year]),
        [group_column, "Year", "Value"],
    ]
    pivot = subset.pivot(index=group_column, columns="Year", values="Value")
    if baseline_year not in pivot or comparison_year not in pivot:
        return pd.DataFrame(columns=[group_column, output_column])
    result = []
    for group, row in pivot.iterrows():
        change = safe_percent_change(
            float(row[comparison_year]),
            float(row[baseline_year]),
        )
        if change is not None:
            result.append({group_column: group, output_column: change})
    return pd.DataFrame(result)


def indexed_to_year(
    values: pd.DataFrame,
    *,
    group_column: str,
    baseline_year: int = 2019,
) -> pd.DataFrame:
    """Index each group's values to 100 in a nonzero baseline year."""
    indexed = values.copy()
    baselines = (
        indexed.loc[indexed["Year"] == baseline_year, [group_column, "Value"]]
        .drop_duplicates(group_column)
        .set_index(group_column)["Value"]
    )
    indexed["Baseline"] = indexed[group_column].map(baselines)
    valid = (
        indexed["Baseline"].notna()
        & np.isfinite(indexed["Baseline"])
        & indexed["Baseline"].ne(0)
    )
    indexed = indexed.loc[valid].copy()
    indexed["Index"] = (indexed["Value"] / indexed["Baseline"]) * 100.0
    return indexed.drop(columns="Baseline").reset_index(drop=True)


def country_travel_receipts(
    data: pd.DataFrame,
    countries: Iterable[str],
    years: Iterable[int],
) -> pd.DataFrame:
    return select_series(
        data,
        report_type="Inbound Tourism-Expenditure",
        metric="Travel",
        countries=countries,
        years=years,
        unique_by_country_year=True,
    )


def transport_comparison(
    data: pd.DataFrame,
    *,
    countries: Iterable[str],
    years: Iterable[int],
) -> pd.DataFrame:
    selected_frames = [
        select_series(
            data,
            report_type="Inbound Tourism-Transport",
            category="Arrivals by mode of transport",
            subcategory="Total",
            metric=metric,
            countries=countries,
            years=years,
            unique_by_country_year=True,
        )
        for metric in ("Air", "Land")
    ]
    return pd.concat(selected_frames, ignore_index=True)


def infrastructure_emptying(data: pd.DataFrame) -> pd.DataFrame:
    receipts = country_travel_receipts(data, ["SPAIN"], [2019, 2020])
    receipts = receipts.assign(Series="Inbound travel receipts")
    overnights = select_series(
        data,
        report_type="Inbound Tourism-Accommodation",
        category="Accommodation",
        subcategory="Hotels and similar establishments",
        metric="Overnights",
        countries=["SPAIN"],
        years=[2019, 2020],
        unique_by_country_year=True,
    ).assign(Series="Inbound hotel overnight stays")
    combined = pd.concat([receipts, overnights], ignore_index=True)
    return indexed_to_year(combined, group_column="Series", baseline_year=2019)


def balance_of_payments(
    data: pd.DataFrame,
    *,
    country: str,
    years: Iterable[int],
) -> pd.DataFrame:
    frames = []
    for report_type, flow in (
        ("Inbound Tourism-Expenditure", "Inbound receipts"),
        ("Outbound Tourism-Expenditure", "Outbound spending"),
    ):
        frame = select_series(
            data,
            report_type=report_type,
            metric="Travel",
            countries=[country],
            years=years,
            unique_by_country_year=True,
        )
        frames.append(frame.assign(Flow=flow))
    return pd.concat(frames, ignore_index=True)


def compute_explorer_kpis(selected: pd.DataFrame) -> ExplorerKpis:
    """Calculate KPIs for one exact series across selected countries."""
    if selected.empty:
        return ExplorerKpis(0.0, None, None, None, None, None)

    yearly = selected.groupby("Year", observed=True)["Value"].sum().sort_index()
    peak_year = int(yearly.idxmax())
    latest_year = int(yearly.index[-1])
    previous_year = int(yearly.index[-2]) if len(yearly) >= 2 else None
    latest_change = None
    if previous_year is not None:
        latest_change = safe_percent_change(
            float(yearly.loc[latest_year]),
            float(yearly.loc[previous_year]),
        )
    return ExplorerKpis(
        cumulative_value=float(selected["Value"].sum()),
        peak_year=peak_year,
        peak_year_value=float(yearly.loc[peak_year]),
        latest_year=latest_year,
        previous_year=previous_year,
        latest_change=latest_change,
    )


def selection_unit(selected: pd.DataFrame) -> str:
    units = sorted(selected["Unit"].dropna().unique())
    if not units:
        return "unknown unit"
    if len(units) > 1:
        raise DataValidationError(f"selection mixes incompatible units: {units}")
    return str(units[0])
