# TSA Data Science: The Global Standstill 🌍📉

### **A Comprehensive Regional Analysis of the COVID-19 Pandemic’s Impact on International Tourism Economics**

**Event:** Data Science and Analytics  
**Team:** Viraat Chauhan & Pranav Sreepada  
**Data Source:** UNWTO (2015-2022)

---

## 📖 Overview

This project is a comprehensive data science portfolio analyzing the catastrophic economic impact of the COVID-19 pandemic on global tourism. Using official UNWTO data, we quantify the "Asymmetric Shock" and "Two-Speed Recovery" across three major regions: **North America, Europe, and Asia**.

The repository includes a **Streamlit Web Application** that serves as an interactive dashboard for exploring the findings, visualizing trends, and drilling down into specific country-level metrics.

---

## 🚀 Features

### **1. Portfolio Report Visuals**
A curated gallery of the 10 key figures used in our written report, interactive and built with Altair:
*   **Baseline & Recovery:** Trends from 2015-2022 comparing pre-pandemic growth vs. post-pandemic recovery.
*   **Economic Impact:** Deep dives into the "Emptying Effect" (Physical vs Financial loss).
*   **Shock & Vulnerability:** Analysis of the 2020 crash (-82% in Asia vs -70% in North America).
*   **Financial Resilience:** Balance of Payments analysis for key economies like Spain.
*   **Case Studies:** Comparative analysis of **USA (Open/Land)** vs **Thailand (Closed/Air)** recovery models.

### **2. Interactive Data Explorer**
A powerful tool for judges and users to validate our findings:
*   **Dynamic Filtering:** Slice data by Country, Metric (Expenditure, Arrivals, etc.), and Year Range.
*   **KPI Dashboard:** Real-time calculation of "Pandemic Impact" (Net Loss), "Market Volatility", and "Recovery Ratio".
*   **Smart Visualizations:** Auto-scaling trend lines with rich tooltips showing YoY growth and precise volumes.
*   **Data Export:** Download filtered datasets as CSV for external verification.

---

## 🛠️ Installation & Setup

Follow these steps to run the dashboard locally on your machine.

### **Prerequisites**
*   Python 3.8 or higher
*   pip (Python package installer)

### **1. Clone the Repository**
```bash
git clone https://github.com/ViraatC22/TSA-Data-Science-Analytics.git
cd TSA-Data-Science-Analytics
```

### **2. Create a Virtual Environment (Optional but Recommended)**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Mac/Linux
# .venv\Scripts\activate   # On Windows
```

### **3. Install Dependencies**
```bash
pip install streamlit pandas altair numpy
```

### **4. Run the Application**
```bash
streamlit run TSA_Data_Science_Analytics.py
```
The app should automatically open in your default browser at `http://localhost:8501`.

---

## 📂 File Structure

```text
TSA-Data-Science-Analytics/
├── TSA_Data_Science_Analytics.py  # Main application code
├── Inbound Tourism-Transport.csv  # Cleaned dataset (Source: UNWTO)
├── README.md                      # Project documentation
└── requirements.txt               # List of python dependencies
```

---

## 📊 Key Findings

1.  **The Asymmetric Shock:** Asia suffered the deepest initial collapse (-82%) due to strict border closures, while North America's open-border policies mitigated the drop to -70%.
2.  **The "Emptying" Effect:** In tourism-dependent economies like Spain, the physical "emptying" of hotels (-81% volume) was more severe than the financial loss (-76% revenue), indicating price floors.
3.  **Two-Speed Recovery:** By 2022, Europe and North America had recovered to ~85% of 2019 levels, while Asia remained stagnant at <40%, confirming a "Two-Speed" global recovery model.

---

## 📜 License

This project is created for the **TSA Data Science and Analytics** competition. All original code and analysis are the intellectual property of the team members. Data is sourced from the UN World Tourism Organization (UNWTO) and is used for educational purposes.
