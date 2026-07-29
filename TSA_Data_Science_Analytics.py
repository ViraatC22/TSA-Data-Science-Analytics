"""Interactive dashboard for the TSA tourism analysis portfolio."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from tsa_analysis import (
    DEFAULT_DATA_PATH,
    DataValidationError,
    REGION_ORDER,
    balance_of_payments,
    compute_explorer_kpis,
    country_travel_receipts,
    grouped_change,
    indexed_to_year,
    infrastructure_emptying,
    load_tourism_data,
    regional_travel_receipts,
    select_series,
    selection_unit,
    transport_comparison,
)


PAGE_CSS = """
    <style>
        .header-container {
            background-color: #ffffff;
            padding: 2.5rem 2rem;
            border-radius: 12px;
            border-left: 8px solid #1E3A8A;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin-bottom: 2rem;
            font-family: Helvetica, Arial, sans-serif;
        }
        .category-tag {
            color: #2563EB !important;
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 0.5rem;
        }
        .main-title {
            color: #000000 !important;
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0 0 1.5rem 0;
            letter-spacing: -1px;
        }
        .meta-data-container {
            display: flex;
            flex-wrap: wrap;
            gap: 2rem;
            border-top: 1px solid #E5E7EB;
            padding-top: 1rem;
            margin-top: 1rem;
        }
        .meta-item { display: flex; flex-direction: column; }
        .meta-label {
            color: #6B7280 !important;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .meta-value {
            color: #374151 !important;
            font-size: 1rem;
            font-weight: 500;
        }
    </style>
    """

REGION_COLORS = ["#2563EB", "#059669", "#D97706"]


def _configure_page() -> None:
    st.set_page_config(
        page_title="TSA Data Science: The Global Standstill",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner="Validating tourism data...")
def load_data() -> pd.DataFrame:
    """Load the repository dataset independently of the current directory."""
    return load_tourism_data(DEFAULT_DATA_PATH)


def _empty_chart(message: str) -> alt.Chart:
    return (
        alt.Chart(pd.DataFrame({"message": [message]}))
        .mark_text(size=16, color="#6B7280")
        .encode(text="message:N")
        .properties(height=300)
    )


def _regional_color(legend: alt.Legend | None = None) -> alt.Color:
    return alt.Color(
        "Region:N",
        scale=alt.Scale(domain=REGION_ORDER, range=REGION_COLORS),
        legend=legend,
    )


def chart_1_baseline(data: pd.DataFrame) -> alt.Chart:
    """Selected-market regional inbound travel receipts."""
    regional = regional_travel_receipts(data)
    if regional.empty:
        return _empty_chart("No regional travel receipt data")
    return (
        alt.Chart(regional)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y(
                "Value:Q",
                title="Inbound travel receipts (USD millions)",
                axis=alt.Axis(format=",.0f"),
            ),
            color=_regional_color(),
            tooltip=[
                "Region:N",
                "Year:O",
                alt.Tooltip("Value:Q", format=",.0f"),
                alt.Tooltip("Country Count:Q"),
            ],
        )
        .properties(
            title="Fig 1: Selected-market inbound travel receipts (2015-2022)",
            height=350,
        )
        .interactive()
    )


def chart_2_shock(data: pd.DataFrame) -> alt.Chart:
    regional = regional_travel_receipts(data, start_year=2019, end_year=2020)
    changes = grouped_change(
        regional,
        group_column="Region",
        baseline_year=2019,
        comparison_year=2020,
        output_column="Change",
    )
    if changes.empty:
        return _empty_chart("2019 and 2020 regional data are required")
    return (
        alt.Chart(changes)
        .mark_bar()
        .encode(
            x=alt.X("Region:N", sort=REGION_ORDER, title=None),
            y=alt.Y(
                "Change:Q",
                title="Change in travel receipts",
                axis=alt.Axis(format="%"),
            ),
            color=_regional_color(legend=None),
            tooltip=["Region:N", alt.Tooltip("Change:Q", format=".1%")],
        )
        .properties(
            title="Fig 2: Descriptive regional shock (2019-2020)",
            height=350,
        )
    )


def chart_3_recovery(data: pd.DataFrame) -> alt.LayerChart:
    regional = regional_travel_receipts(data, start_year=2019, end_year=2022)
    indexed = indexed_to_year(
        regional,
        group_column="Region",
        baseline_year=2019,
    )
    if indexed.empty:
        return _empty_chart("A nonzero 2019 baseline is required")
    line = (
        alt.Chart(indexed)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("Index:Q", title="Receipt index (2019 = 100)"),
            color=_regional_color(),
            tooltip=[
                "Region:N",
                "Year:O",
                alt.Tooltip("Index:Q", format=".1f"),
                alt.Tooltip("Country Count:Q"),
            ],
        )
    )
    baseline = (
        alt.Chart(pd.DataFrame({"Index": [100]}))
        .mark_rule(color="#6B7280", strokeDash=[5, 5])
        .encode(y="Index:Q")
    )
    return (line + baseline).properties(
        title="Fig 3: Selected-market recovery relative to 2019",
        height=350,
    )


def chart_4_transport_vulnerability(data: pd.DataFrame) -> alt.Chart:
    transport = transport_comparison(
        data,
        countries=["SPAIN", "MEXICO"],
        years=[2019, 2020],
    )
    if transport.empty:
        return _empty_chart("No comparable air and land series")
    return (
        alt.Chart(transport)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title="Transport mode"),
            y=alt.Y(
                "Value:Q",
                title="Arrivals (thousands)",
                axis=alt.Axis(format=",.0f"),
            ),
            color=alt.Color("Year:O", scale=alt.Scale(scheme="blues")),
            column=alt.Column("Country:N", title=None),
            tooltip=[
                "Country:N",
                "Year:O",
                "Metric:N",
                alt.Tooltip("Value:Q", format=",.1f"),
            ],
        )
        .properties(
            title="Fig 4: Air and land arrivals (2019 vs. 2020)",
            height=300,
            width=200,
        )
    )


def chart_5_infrastructure_emptying(data: pd.DataFrame) -> alt.Chart:
    emptying = infrastructure_emptying(data)
    if emptying.empty:
        return _empty_chart("Spain receipt and accommodation baselines are required")
    return (
        alt.Chart(emptying)
        .mark_line(strokeWidth=4, point={"size": 100, "filled": True})
        .encode(
            x=alt.X("Year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "Index:Q",
                title="Index (2019 = 100)",
                scale=alt.Scale(domain=[0, 105]),
            ),
            color=alt.Color(
                "Series:N",
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                "Series:N",
                "Year:O",
                alt.Tooltip("Index:Q", format=".1f"),
                "Unit:N",
            ],
        )
        .properties(
            title="Fig 5: Spain receipts vs. inbound hotel overnight stays",
            height=350,
        )
    )


def chart_6_balance_payments(data: pd.DataFrame) -> alt.Chart:
    flows = balance_of_payments(data, country="SPAIN", years=[2019, 2020])
    if flows.empty:
        return _empty_chart("Spain inbound and outbound receipt data are required")
    return (
        alt.Chart(flows)
        .mark_bar()
        .encode(
            x=alt.X("Flow:N", title=None),
            y=alt.Y(
                "Value:Q",
                title="Travel receipts/spending (USD millions)",
                axis=alt.Axis(format=",.0f"),
            ),
            color=alt.Color(
                "Flow:N",
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            column=alt.Column("Year:O", title=None),
            tooltip=[
                "Year:O",
                "Flow:N",
                alt.Tooltip("Value:Q", format=",.0f"),
            ],
        )
        .properties(
            title="Fig 6: Spain inbound and outbound travel flows",
            height=300,
            width=170,
        )
    )


def chart_7_absolute_loss(data: pd.DataFrame) -> alt.Chart:
    regional = regional_travel_receipts(data, start_year=2019, end_year=2020)
    pivot = regional.pivot(index="Region", columns="Year", values="Value")
    if 2019 not in pivot or 2020 not in pivot:
        return _empty_chart("2019 and 2020 regional data are required")
    losses = (pivot[2019] - pivot[2020]).rename("Absolute Loss").reset_index()
    return (
        alt.Chart(losses)
        .mark_bar()
        .encode(
            x=alt.X("Region:N", sort="-y", title=None),
            y=alt.Y(
                "Absolute Loss:Q",
                title="Receipt loss (USD millions)",
                axis=alt.Axis(format=",.0f"),
            ),
            color=_regional_color(legend=None),
            tooltip=[
                "Region:N",
                alt.Tooltip("Absolute Loss:Q", format=",.0f"),
            ],
        )
        .properties(
            title="Fig 7: Absolute selected-market receipt loss (2019-2020)",
            height=350,
        )
    )


def chart_8_top10_drops(data: pd.DataFrame) -> alt.Chart:
    receipts = select_series(
        data,
        report_type="Inbound Tourism-Expenditure",
        metric="Travel",
        regions=REGION_ORDER,
        years=[2019, 2020],
        unique_by_country_year=True,
    )
    changes = grouped_change(
        receipts,
        group_column="Country",
        baseline_year=2019,
        comparison_year=2020,
        output_column="Change",
    )
    if changes.empty:
        return _empty_chart("Comparable country receipt data are unavailable")
    hardest_hit = changes.sort_values("Change").head(10)
    return (
        alt.Chart(hardest_hit)
        .mark_bar()
        .encode(
            x=alt.X("Change:Q", title="Change", axis=alt.Axis(format="%")),
            y=alt.Y("Country:N", sort="x", title=None),
            color=alt.Color(
                "Change:Q",
                scale=alt.Scale(scheme="reds", reverse=True),
                legend=None,
            ),
            tooltip=["Country:N", alt.Tooltip("Change:Q", format=".1%")],
        )
        .properties(
            title="Fig 8: Largest selected-country receipt declines (2020)",
            height=400,
        )
    )


def chart_9_recovery_velocity(data: pd.DataFrame) -> alt.Chart:
    regional = regional_travel_receipts(data, start_year=2021, end_year=2022)
    changes = grouped_change(
        regional,
        group_column="Region",
        baseline_year=2021,
        comparison_year=2022,
        output_column="Change",
    )
    if changes.empty:
        return _empty_chart("2021 and 2022 regional data are required")
    return (
        alt.Chart(changes)
        .mark_bar()
        .encode(
            x=alt.X("Region:N", sort="-y", title=None),
            y=alt.Y(
                "Change:Q",
                title="Travel receipt change (2021-2022)",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "Change:Q",
                scale=alt.Scale(scheme="greens"),
                legend=None,
            ),
            tooltip=["Region:N", alt.Tooltip("Change:Q", format=".1%")],
        )
        .properties(
            title="Fig 9: Selected-market recovery momentum",
            height=350,
        )
    )


def chart_10_us_vs_thailand(data: pd.DataFrame) -> alt.Chart:
    countries = ["UNITED STATES OF AMERICA", "THAILAND"]
    receipts = country_travel_receipts(data, countries, range(2019, 2023))
    indexed = indexed_to_year(
        receipts,
        group_column="Country",
        baseline_year=2019,
    )
    if indexed.empty:
        return _empty_chart("Country receipt baselines are unavailable")
    return (
        alt.Chart(indexed)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("Index:Q", title="Travel receipt index (2019 = 100)"),
            color=alt.Color("Country:N"),
            tooltip=[
                "Country:N",
                "Year:O",
                alt.Tooltip("Index:Q", format=".1f"),
            ],
        )
        .properties(
            title="Fig 10: United States and Thailand recovery paths",
            height=350,
        )
    )


def _index_or_zero(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


def render_portfolio(data: pd.DataFrame) -> None:
    st.subheader("Portfolio report visuals")
    st.info(
        "Method note: regional figures are descriptive sums over the selected "
        "countries in the project taxonomy. They use only the inbound "
        "`Travel` receipt series (USD millions), avoiding overlap with `Total`."
    )

    regional = regional_travel_receipts(data, start_year=2019, end_year=2022)
    shock = grouped_change(
        regional,
        group_column="Region",
        baseline_year=2019,
        comparison_year=2020,
        output_column="Change",
    ).set_index("Region")
    recovery = indexed_to_year(
        regional,
        group_column="Region",
        baseline_year=2019,
    )
    recovery_2022 = recovery.loc[recovery["Year"] == 2022].set_index("Region")
    metric_columns = st.columns(3)
    for column, region in zip(metric_columns, REGION_ORDER, strict=True):
        drop_value = shock.at[region, "Change"] if region in shock.index else None
        recovery_value = (
            recovery_2022.at[region, "Index"]
            if region in recovery_2022.index
            else None
        )
        with column:
            st.metric(
                region,
                f"{recovery_value:.1f}% of 2019"
                if recovery_value is not None
                else "Unavailable",
                f"{drop_value:.1%} in 2020" if drop_value is not None else None,
                delta_color="normal",
            )

    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Baseline and recovery")
        st.altair_chart(chart_1_baseline(data), width="stretch")
        st.altair_chart(chart_3_recovery(data), width="stretch")
        st.markdown("#### Economic impact")
        st.altair_chart(
            chart_5_infrastructure_emptying(data),
            width="stretch",
        )
        st.altair_chart(chart_7_absolute_loss(data), width="stretch")
        st.markdown("#### Recovery momentum")
        st.altair_chart(
            chart_9_recovery_velocity(data),
            width="stretch",
        )
    with right:
        st.markdown("#### Shock and transport")
        st.altair_chart(chart_2_shock(data), width="stretch")
        st.altair_chart(
            chart_4_transport_vulnerability(data),
            width="stretch",
        )
        st.markdown("#### Financial flows")
        st.altair_chart(
            chart_6_balance_payments(data),
            width="stretch",
        )
        st.altair_chart(chart_8_top10_drops(data), width="stretch")
        st.markdown("#### Case study")
        st.altair_chart(
            chart_10_us_vs_thailand(data),
            width="stretch",
        )


def render_explorer(data: pd.DataFrame) -> None:
    st.subheader("Validated data explorer")
    st.markdown(
        "Choose one complete statistical series. The controls intentionally "
        "prevent expenditure, arrival, accommodation, and employment units "
        "from being combined."
    )

    report_options = sorted(data["Report Type"].unique())
    report_type = st.selectbox(
        "Report type",
        report_options,
        index=_index_or_zero(
            report_options,
            "Inbound Tourism-Expenditure",
        ),
    )
    report_data = data.loc[data["Report Type"] == report_type]

    control_columns = st.columns(3)
    category_options = sorted(report_data["Category"].unique())
    with control_columns[0]:
        category = st.selectbox("Category", category_options)
    category_data = report_data.loc[report_data["Category"] == category]

    subcategory_options = sorted(category_data["Subcategory"].unique())
    with control_columns[1]:
        subcategory = st.selectbox(
            "Subcategory",
            subcategory_options,
            index=_index_or_zero(subcategory_options, "Travel"),
        )
    subcategory_data = category_data.loc[
        category_data["Subcategory"] == subcategory
    ]

    metric_options = sorted(subcategory_data["Metric"].unique())
    with control_columns[2]:
        metric = st.selectbox(
            "Metric",
            metric_options,
            index=_index_or_zero(metric_options, "Travel"),
        )

    exact_series = select_series(
        data,
        report_type=report_type,
        category=category,
        subcategory=subcategory,
        metric=metric,
        unique_by_country_year=True,
    )
    if exact_series.empty:
        st.warning("No observations are available for this exact series.")
        return

    unit = selection_unit(exact_series)
    country_options = sorted(exact_series["Country"].unique())
    preferred = [
        "SPAIN",
        "THAILAND",
        "UNITED STATES OF AMERICA",
        "FRANCE",
    ]
    default_countries = [
        country for country in preferred if country in country_options
    ]
    if not default_countries:
        default_countries = country_options[: min(4, len(country_options))]

    countries = st.multiselect(
        "Countries",
        country_options,
        default=default_countries,
    )
    min_year = int(exact_series["Year"].min())
    max_year = int(exact_series["Year"].max())
    default_years = (max(min_year, 2015), min(max_year, 2022))
    if default_years[0] > default_years[1]:
        default_years = (min_year, max_year)
    year_range = st.slider(
        "Year range",
        min_year,
        max_year,
        default_years,
    )

    selected = exact_series.loc[
        exact_series["Country"].isin(countries)
        & exact_series["Year"].between(*year_range)
    ].copy()
    if selected.empty:
        st.warning("No observations match the selected countries and years.")
        return

    st.caption(
        f"Unit: **{unit}** · {selected['Country'].nunique()} countries · "
        f"{len(selected):,} observations"
    )
    kpis = compute_explorer_kpis(selected)
    kpi_columns = st.columns(3)
    kpi_columns[0].metric(
        "Cumulative selected value",
        f"{kpis.cumulative_value:,.1f}",
        help=f"Sum across selected observations; unit: {unit}",
    )
    kpi_columns[1].metric(
        f"Peak aggregate year ({kpis.peak_year})",
        f"{kpis.peak_year_value:,.1f}",
        help=f"Sum across selected countries; unit: {unit}",
    )
    if kpis.latest_change is None:
        trend_label = "Latest comparable change"
        trend_value = "Not available"
    else:
        trend_label = f"Change ({kpis.latest_year} vs. {kpis.previous_year})"
        trend_value = f"{kpis.latest_change:+.1%}"
    kpi_columns[2].metric(trend_label, trend_value)

    chart = (
        alt.Chart(selected)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "Value:Q",
                title=unit,
                axis=alt.Axis(format=",.1f"),
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "Country:N",
                legend=alt.Legend(title="Country", orient="bottom"),
            ),
            tooltip=[
                "Country:N",
                "Year:O",
                alt.Tooltip("Value:Q", format=",.1f"),
                "Unit:N",
            ],
        )
        .interactive()
    )
    st.altair_chart(chart, width="stretch")

    table = selected.pivot(index="Country", columns="Year", values="Value")
    st.dataframe(table, width="stretch")
    csv_columns = [
        "Country",
        "Report Type",
        "Category",
        "Subcategory",
        "Metric",
        "Year",
        "Value",
        "Unit",
    ]
    st.download_button(
        label="Download selected observations as CSV",
        data=selected[csv_columns].to_csv(index=False).encode("utf-8"),
        file_name="tsa_selected_data.csv",
        mime="text/csv",
    )


def main() -> None:
    _configure_page()
    with st.sidebar:
        st.title("TSA Data Science")
        st.caption("Category: Data Science and Analytics")
        st.caption("Team: Viraat Chauhan & Pranav Sreepada")
        st.divider()
        mode = st.radio(
            "Select module",
            ["Portfolio Report Visuals", "Interactive Data Explorer"],
        )
        st.divider()
        st.info("**Project title:**\nThe Global Standstill")
        with st.expander("Method and limitations"):
            st.markdown(
                "- Source coverage: 1995-2022\n"
                "- Main regional scope: selected markets, not complete regions\n"
                "- Main measure: inbound `Travel` receipts, current USD millions\n"
                "- Analysis: descriptive; no causal or forecast model\n"
                "- Counts are shown in thousands when documented upstream"
            )

    try:
        data = load_data()
    except (FileNotFoundError, DataValidationError) as error:
        st.error(f"Unable to load validated tourism data: {error}")
        st.stop()

    st.markdown(
        """
        <div class="header-container">
            <p class="category-tag">TSA Data Science & Analytics Portfolio</p>
            <h1 class="main-title">THE GLOBAL STANDSTILL</h1>
            <div class="meta-data-container">
                <div class="meta-item">
                    <span class="meta-label">Team</span>
                    <span class="meta-value">Viraat Chauhan & Pranav Sreepada</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Scope</span>
                    <span class="meta-value">Descriptive COVID-19 tourism analysis</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Data</span>
                    <span class="meta-value">UN Tourism-derived, 1995-2022</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if mode == "Portfolio Report Visuals":
        render_portfolio(data)
    else:
        render_explorer(data)

    st.caption(
        "Aggregate country-level research data. Source, license, coverage, and "
        "methodology limitations are documented in the repository."
    )


if __name__ == "__main__":
    main()
