from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from tsa_analysis import (
    DEFAULT_DATA_PATH,
    DataValidationError,
    balance_of_payments,
    compute_explorer_kpis,
    grouped_change,
    indexed_to_year,
    infrastructure_emptying,
    load_tourism_data,
    regional_travel_receipts,
    safe_percent_change,
    select_series,
    selection_unit,
    transport_comparison,
)


class SourceValidationTests(unittest.TestCase):
    def _write_source(
        self,
        directory: Path,
        rows: list[dict[str, object]],
    ) -> Path:
        path = directory / "source.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_loader_normalizes_supported_thousands_separators(self) -> None:
        with TemporaryDirectory() as temp:
            source = self._write_source(
                Path(temp),
                [
                    {
                        "Country": "Example",
                        "Report Type": "Domestic Tourism-Trips",
                        "Category": "Trips",
                        "Subcategory": "Total trips",
                        "Metric": "Same-day visitors (excursionists)",
                        "2021": "1,234",
                        "2022": "46\u00a0264",
                    }
                ],
            )
            loaded = load_tourism_data(source)
            self.assertEqual(loaded["Value"].tolist(), [1234.0, 46264.0])
            self.assertEqual(loaded["Country"].unique().tolist(), ["EXAMPLE"])

    def test_loader_rejects_bad_schema_duplicates_and_tokens(self) -> None:
        base = {
            "Country": "Example",
            "Report Type": "Inbound Tourism-Expenditure",
            "Category": "Tourism expenditure in the country",
            "Subcategory": "Travel",
            "Metric": "Travel",
            "2022": "100",
        }
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            missing = dict(base)
            missing.pop("Metric")
            with self.assertRaisesRegex(DataValidationError, "missing required"):
                load_tourism_data(self._write_source(directory, [missing]))

            with self.assertRaisesRegex(DataValidationError, "duplicate"):
                load_tourism_data(
                    self._write_source(directory, [base, base.copy()])
                )

            invalid = dict(base)
            invalid["2022"] = "not reported"
            with self.assertRaisesRegex(DataValidationError, "non-numeric"):
                load_tourism_data(self._write_source(directory, [invalid]))


class RealDatasetRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_tourism_data(DEFAULT_DATA_PATH)

    def test_real_dataset_shape_range_and_formatted_italy_value(self) -> None:
        self.assertEqual(len(self.data), 91514)
        self.assertEqual(
            (int(self.data["Year"].min()), int(self.data["Year"].max())),
            (1995, 2022),
        )
        italy = self.data.loc[
            (self.data["Country"] == "ITALY")
            & (self.data["Report Type"] == "Domestic Tourism-Trips")
            & (self.data["Metric"] == "Same-day visitors (excursionists)")
            & (self.data["Year"] == 2022)
        ]
        self.assertEqual(italy["Value"].tolist(), [46264.0])

    def test_regional_receipts_use_one_series_and_match_source(self) -> None:
        regional = regional_travel_receipts(
            self.data,
            start_year=2019,
            end_year=2022,
        )
        pivot = regional.pivot(index="Region", columns="Year", values="Value")
        self.assertEqual(float(pivot.loc["Asia", 2019]), 209155.0)
        self.assertEqual(float(pivot.loc["Europe", 2020]), 147215.0)
        self.assertEqual(float(pivot.loc["North America", 2022]), 189128.0)
        self.assertTrue(regional["Unit"].eq("USD millions").all())

        changes = grouped_change(
            regional,
            group_column="Region",
            baseline_year=2019,
            comparison_year=2020,
        ).set_index("Region")
        self.assertAlmostEqual(changes.at["Asia", "Change"], -0.7448734192)
        self.assertAlmostEqual(changes.at["Europe", "Change"], -0.5869712817)

        recovery = indexed_to_year(
            regional,
            group_column="Region",
            baseline_year=2019,
        )
        europe_2022 = recovery.loc[
            (recovery["Region"] == "Europe") & (recovery["Year"] == 2022),
            "Index",
        ].item()
        self.assertAlmostEqual(europe_2022, 93.4584263863, places=6)

    def test_mechanism_series_have_explicit_units_and_expected_values(self) -> None:
        transport = transport_comparison(
            self.data,
            countries=["SPAIN", "MEXICO"],
            years=[2019, 2020],
        )
        self.assertEqual(len(transport), 8)
        self.assertEqual(selection_unit(transport), "thousand arrivals")
        spain_air_2020 = transport.loc[
            (transport["Country"] == "SPAIN")
            & (transport["Metric"] == "Air")
            & (transport["Year"] == 2020),
            "Value",
        ].item()
        self.assertEqual(spain_air_2020, 13657.7)

        emptying = infrastructure_emptying(self.data)
        hotel_2020 = emptying.loc[
            (emptying["Series"] == "Inbound hotel overnight stays")
            & (emptying["Year"] == 2020),
            "Index",
        ].item()
        self.assertAlmostEqual(hotel_2020, 18.3861119318, places=6)

        flows = balance_of_payments(
            self.data,
            country="SPAIN",
            years=[2019, 2020],
        )
        inbound_2019 = flows.loc[
            (flows["Flow"] == "Inbound receipts") & (flows["Year"] == 2019),
            "Value",
        ].item()
        self.assertEqual(inbound_2019, 79571.0)

    def test_ambiguous_series_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "ambiguous"):
            select_series(
                self.data,
                report_type="Inbound Tourism-Accommodation",
                metric="Overnights",
                unique_by_country_year=True,
            )


class SafeStatisticsTests(unittest.TestCase):
    def test_percent_change_rejects_zero_and_nonfinite_baselines(self) -> None:
        self.assertIsNone(safe_percent_change(1.0, 0.0))
        self.assertIsNone(safe_percent_change(1.0, np.nan))
        self.assertEqual(safe_percent_change(120.0, 100.0), 0.2)

    def test_kpis_handle_empty_single_year_and_zero_previous_year(self) -> None:
        empty = compute_explorer_kpis(pd.DataFrame(columns=["Year", "Value"]))
        self.assertEqual(empty.cumulative_value, 0.0)
        self.assertIsNone(empty.latest_change)

        selected = pd.DataFrame(
            {
                "Year": [2020, 2021, 2021],
                "Value": [0.0, 10.0, 5.0],
            }
        )
        kpis = compute_explorer_kpis(selected)
        self.assertEqual(kpis.peak_year, 2021)
        self.assertEqual(kpis.peak_year_value, 15.0)
        self.assertIsNone(kpis.latest_change)


if __name__ == "__main__":
    unittest.main()
