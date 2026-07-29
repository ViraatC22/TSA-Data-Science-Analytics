from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

import TSA_Data_Science_Analytics as dashboard
from tsa_analysis import load_tourism_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "TSA_Data_Science_Analytics.py"


class ChartConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_tourism_data()

    def test_all_portfolio_charts_compile_to_nonempty_vega_specs(self) -> None:
        chart_functions = [
            dashboard.chart_1_baseline,
            dashboard.chart_2_shock,
            dashboard.chart_3_recovery,
            dashboard.chart_4_transport_vulnerability,
            dashboard.chart_5_infrastructure_emptying,
            dashboard.chart_6_balance_payments,
            dashboard.chart_7_absolute_loss,
            dashboard.chart_8_top10_drops,
            dashboard.chart_9_recovery_velocity,
            dashboard.chart_10_us_vs_thailand,
        ]
        for function in chart_functions:
            with self.subTest(chart=function.__name__):
                specification = function(self.data).to_dict()
                self.assertIn("title", specification)
                self.assertGreater(len(str(specification)), 500)


class StreamlitFlowTests(unittest.TestCase):
    def test_portfolio_and_explorer_render_without_exceptions(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(
            app.radio[0].value,
            "Portfolio Report Visuals",
        )
        self.assertGreaterEqual(len(app.metric), 3)

        app.radio[0].set_value("Interactive Data Explorer")
        app.run()
        self.assertEqual(list(app.exception), [])
        self.assertGreaterEqual(len(app.selectbox), 3)
        self.assertEqual(
            app.selectbox[0].value,
            "Inbound Tourism-Expenditure",
        )
        self.assertEqual(app.selectbox[2].value, "Travel")
        self.assertGreaterEqual(len(app.metric), 3)


if __name__ == "__main__":
    unittest.main()
