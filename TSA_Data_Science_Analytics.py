import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TSA Data Science: The Global Standstill",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
<style>
    /* 1. HERO BANNER CONTAINER */
    .header-container {
        background-color: #ffffff;
        padding: 2.5rem 2rem;
        border-radius: 12px;
        border-left: 8px solid #1E3A8A; /* Professional Blue Accent */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        font-family: 'Helvetica', 'Arial', sans-serif;
    }

    /* 2. CATEGORY TAG (Top Label) */
    .category-tag {
        color: #2563EB !important; /* Lighter Blue - Forced */
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }

    /* 3. MAIN TITLE */
    .main-title {
        color: #000000 !important; /* Black - Forced */
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0 0 1.5rem 0;
        letter-spacing: -1px;
    }

    /* 4. METADATA SECTION (Team ID, etc.) */
    .meta-data-container {
        display: flex;
        flex-wrap: wrap;
        gap: 2rem;
        border-top: 1px solid #E5E7EB;
        padding-top: 1rem;
        margin-top: 1rem;
    }

    .meta-item {
        display: flex;
        flex-direction: column;
    }

    .meta-label {
        color: #6B7280 !important; /* Grey - Forced */
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .meta-value {
        color: #374151 !important; /* Dark Grey - Forced */
        font-size: 1rem;
        font-weight: 500;
    }

    /* TAB STYLING */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom: 2px solid #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING & PROCESSING ---
@st.cache_data
def load_data():
    """
    Loads and processes the 'Inbound Tourism-Transport.csv' file.
    """
    file_path = 'Inbound Tourism-Transport.csv'
    
    try:
        df = pd.read_csv(file_path)
        
        # Identify Year columns
        year_cols = [col for col in df.columns if col.isdigit()]
        
        # Melt
        id_vars = ['Country', 'Report Type', 'Category', 'Subcategory', 'Metric']
        # Filter id_vars to only those present in the CSV
        available_id_vars = [col for col in id_vars if col in df.columns]
        
        df_long = pd.melt(
            df, 
            id_vars=available_id_vars, 
            value_vars=year_cols, 
            var_name='Year', 
            value_name='Value'
        )
        
        # Clean Data
        df_long['Year'] = pd.to_numeric(df_long['Year'])
        df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
        df_long = df_long.dropna(subset=['Value'])
        df_long['Country'] = df_long['Country'].str.upper().str.strip()
        
        return df_long

    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure the CSV is in the same directory.")
        return pd.DataFrame()

# --- 2. REGIONAL MAPPING ---
REGION_MAP = {
    # North America
    'UNITED STATES': 'North America', 'USA': 'North America', 'UNITED STATES OF AMERICA': 'North America',
    'CANADA': 'North America', 'MEXICO': 'North America',
    
    # Europe
    'FRANCE': 'Europe', 'SPAIN': 'Europe', 'ITALY': 'Europe', 'GERMANY': 'Europe',
    'UNITED KINGDOM': 'Europe', 'UK': 'Europe', 'AUSTRIA': 'Europe', 'GREECE': 'Europe', 'PORTUGAL': 'Europe',
    
    # Asia
    'THAILAND': 'Asia', 'CHINA': 'Asia', 'JAPAN': 'Asia', 'REPUBLIC OF KOREA': 'Asia', 'SOUTH KOREA': 'Asia',
    'INDIA': 'Asia', 'INDONESIA': 'Asia', 'MALAYSIA': 'Asia', 'VIET NAM': 'Asia'
}

def add_regions(df):
    df['Region'] = df['Country'].map(REGION_MAP)
    return df

# --- 3. CHART GENERATION FUNCTIONS ---

def chart_1_baseline(df):
    """Fig 1: Regional Inbound Expenditure"""
    df_exp = df[
        (df['Report Type'].str.contains('Inbound Tourism-Expenditure', na=False)) &
        (df['Metric'].isin(['Travel', 'Total'])) &
        (df['Region'].notna()) &
        (df['Year'].between(2015, 2022))
    ]
    df_agg = df_exp.groupby(['Region', 'Year'])['Value'].sum().reset_index()
    
    return alt.Chart(df_agg).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('Year:O', title='Year'),
        # Fix: format inside Axis
        y=alt.Y('Value:Q', title='Expenditure (USD Millions)', axis=alt.Axis(format=',.0f')),
        color=alt.Color('Region', scale=alt.Scale(scheme='category10')),
        tooltip=['Region', 'Year', alt.Tooltip('Value', format=',.0f')]
    ).properties(
        title='Fig 1: Regional Inbound Expenditure (2015-2022) - The Pre-Pandemic Baseline',
        height=350
    ).interactive()

def chart_2_shock(df):
    """Fig 2: % Drop"""
    df_exp = df[
        (df['Report Type'].str.contains('Inbound Tourism-Expenditure', na=False)) &
        (df['Metric'].isin(['Travel', 'Total'])) &
        (df['Region'].notna()) &
        (df['Year'].isin([2019, 2020]))
    ]
    df_agg = df_exp.groupby(['Region', 'Year'])['Value'].sum().reset_index()
    df_pivot = df_agg.pivot(index='Region', columns='Year', values='Value').reset_index()
    df_pivot['% Drop'] = ((df_pivot[2020] - df_pivot[2019]) / df_pivot[2019])
    
    return alt.Chart(df_pivot).mark_bar().encode(
        x=alt.X('Region', sort=['North America', 'Europe', 'Asia']),
        y=alt.Y('% Drop', axis=alt.Axis(format='%')),
        color=alt.Color('Region', legend=None),
        tooltip=['Region', alt.Tooltip('% Drop', format='.1%')]
    ).properties(
        title='Fig 2: The Asymmetric Shock (% Drop 2019-2020)',
        height=350
    )

def chart_3_recovery(df):
    """Fig 3: Recovery Index"""
    df_exp = df[
        (df['Report Type'].str.contains('Inbound Tourism-Expenditure', na=False)) &
        (df['Metric'].isin(['Travel', 'Total'])) &
        (df['Region'].notna()) &
        (df['Year'].between(2019, 2022))
    ]
    df_agg = df_exp.groupby(['Region', 'Year'])['Value'].sum().reset_index()
    df_agg['Index'] = 0.0
    for region in df_agg['Region'].unique():
        try:
            base_val = df_agg[(df_agg['Region'] == region) & (df_agg['Year'] == 2019)]['Value'].values[0]
            df_agg.loc[df_agg['Region'] == region, 'Index'] = (df_agg['Value'] / base_val) * 100
        except IndexError: continue

    line = alt.Chart(df_agg).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('Year:O'),
        y=alt.Y('Index', title='Index (2019 = 100)'),
        color='Region',
        tooltip=['Region', 'Year', alt.Tooltip('Index', format='.1f')]
    )
    rule = alt.Chart(pd.DataFrame({'y': [100]})).mark_rule(color='gray', strokeDash=[5,5]).encode(y='y')
    return (line + rule).properties(title='Fig 3: The Two-Speed Recovery (2019 Baseline = 100)', height=350)

def chart_4_transport_vulnerability(df):
    """Fig 4: Air vs Land"""
    target_countries = ['SPAIN', 'MEXICO']
    df_trans = df[
        (df['Report Type'].str.contains('Transport', na=False)) &
        (df['Metric'].isin(['Air', 'Land'])) &
        (df['Country'].isin(target_countries)) &
        (df['Year'].isin([2019, 2020]))
    ]
    if df_trans.empty: return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("No Data"))

    return alt.Chart(df_trans).mark_bar().encode(
        x=alt.X('Metric', title='Transport Mode'),
        # Fix: format inside Axis
        y=alt.Y('Value', title='Arrivals', axis=alt.Axis(format=',.0f')),
        color='Year:O',
        column=alt.Column('Country', title=None),
        tooltip=['Country', 'Year', 'Metric', alt.Tooltip('Value', format=',.0f')]
    ).properties(
        title='Fig 4: Mechanism of Impact - Air vs. Land Resilience (2019 vs 2020)',
        height=300, width=200
    )

def chart_5_infrastructure_emptying(df):
    """Fig 5: Emptying Effect"""
    df_spain = df[(df['Country'] == 'SPAIN') & (df['Year'].isin([2019, 2020]))].copy()
    exp = df_spain[(df_spain['Report Type'].str.contains('Inbound Tourism-Expenditure')) & (df_spain['Metric']=='Travel')].copy()
    exp['Metric_Label'] = 'Expenditure ($)'
    hotel = df_spain[(df_spain['Report Type'].str.contains('Accommodation')) & (df_spain['Metric']=='Overnights') & (df_spain['Subcategory'].str.contains('Hotels', na=False))].copy()
    hotel['Metric_Label'] = 'Hotel Overnights'
    
    combined = pd.concat([exp, hotel])
    combined['Index'] = 0.0
    for m in combined['Metric_Label'].unique():
        subset = combined[(combined['Metric_Label']==m) & (combined['Year']==2019)]
        if not subset.empty:
            base = subset['Value'].values[0]
            combined.loc[combined['Metric_Label']==m, 'Index'] = (combined['Value'] / base) * 100
    combined = combined[(combined['Index'] >= 0) & (combined['Index'] <= 110)]
    
    if combined.empty: return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("No Data"))

    base = alt.Chart(combined).encode(
        x=alt.X('Year:O', scale=alt.Scale(padding=0.4), axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y('Index:Q', title='Normalized Volume (2019 = 100)', scale=alt.Scale(domain=[0, 105])),
        color=alt.Color('Metric_Label:N', legend=alt.Legend(title=None, orient='bottom'))
    )
    lines = base.mark_line(strokeWidth=4, point={'size': 100, 'filled': True}).encode(
        tooltip=['Metric_Label', 'Year', alt.Tooltip('Index', format='.1f')]
    )
    return lines.properties(title='Fig 5: The "Emptying" Effect (Spain) - Financial vs. Physical Drop', height=350)

def chart_6_balance_payments(df):
    """Fig 6: Balance of Payments"""
    df_spain = df[
        (df['Country'] == 'SPAIN') & 
        (df['Metric'] == 'Travel') &
        (df['Report Type'].isin(['Inbound Tourism-Expenditure', 'Outbound Tourism-Expenditure'])) &
        (df['Year'].isin([2019, 2020]))
    ].copy()
    df_spain['Flow'] = df_spain['Report Type'].apply(lambda x: 'Inbound (Earnings)' if 'Inbound' in x else 'Outbound (Spending)')
    
    return alt.Chart(df_spain).mark_bar().encode(
        x=alt.X('Flow', title=None),
        # Fix: format inside Axis
        y=alt.Y('Value', title='USD Millions', axis=alt.Axis(format=',.0f')),
        color=alt.Color('Flow', legend=alt.Legend(title=None, orient='bottom')),
        column='Year:O',
        tooltip=['Year', 'Flow', alt.Tooltip('Value', format=',.0f')]
    ).properties(title='Fig 6: Economic Buffer - Balance of Payments (Spain)', height=300, width=150)

def chart_7_absolute_loss(df):
    """Fig 7: Absolute Loss"""
    df_exp = df[
        (df['Report Type'].str.contains('Inbound Tourism-Expenditure', na=False)) &
        (df['Metric'].isin(['Travel', 'Total'])) &
        (df['Region'].notna()) &
        (df['Year'].isin([2019, 2020]))
    ]
    df_agg = df_exp.groupby(['Region', 'Year'])['Value'].sum().unstack()
    df_loss = (df_agg[2019] - df_agg[2020]).reset_index(name='Absolute Loss')
    
    return alt.Chart(df_loss).mark_bar().encode(
        x=alt.X('Region', sort='-y'),
        # Fix: format inside Axis
        y=alt.Y('Absolute Loss', title='Loss in USD Millions', axis=alt.Axis(format=',.0f')),
        color=alt.Color('Region'),
        tooltip=[alt.Tooltip('Absolute Loss', format=',.0f')]
    ).properties(title='Fig 7: Absolute Economic Loss (2019 -> 2020)', height=350)

def chart_8_top10_drops(df):
    """Fig 8: Top 10 Drops"""
    df_exp = df[
        (df['Report Type'].str.contains('Inbound Tourism-Expenditure', na=False)) &
        (df['Metric'].isin(['Travel', 'Total'])) &
        (df['Region'].notna()) &
        (df['Year'].isin([2019, 2020]))
    ]
    pivot = df_exp.pivot_table(index='Country', columns='Year', values='Value').reset_index()
    if 2019 in pivot.columns and 2020 in pivot.columns:
        pivot = pivot.dropna()
        pivot['% Drop'] = (pivot[2020] - pivot[2019]) / pivot[2019]
        top10 = pivot.sort_values('% Drop').head(10)
        return alt.Chart(top10).mark_bar().encode(
            x=alt.X('% Drop', axis=alt.Axis(format='%')),
            y=alt.Y('Country', sort='x'),
            color=alt.Color('% Drop', scale=alt.Scale(scheme='reds', reverse=True)),
            tooltip=['Country', alt.Tooltip('% Drop', format='.1%')]
        ).properties(title='Fig 8: The Hardest Hit - Top 10 Countries by % Drop (2020)', height=400)
    return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("Insufficient Data"))

def chart_9_recovery_velocity(df):
    """Fig 9: Recovery Velocity"""
    df_exp = df[
        (df['Report Type'].str.contains('Inbound Tourism-Expenditure', na=False)) &
        (df['Metric'].isin(['Travel', 'Total'])) &
        (df['Region'].notna()) &
        (df['Year'].isin([2021, 2022]))
    ]
    df_agg = df_exp.groupby(['Region', 'Year'])['Value'].sum().unstack()
    df_velocity = ((df_agg[2022] - df_agg[2021]) / df_agg[2021]).reset_index(name='Growth_Rate')
    
    return alt.Chart(df_velocity).mark_bar().encode(
        x=alt.X('Region', sort='-y'),
        # Fix: format inside Axis
        y=alt.Y('Growth_Rate', axis=alt.Axis(format='%', title='YoY Growth Rate (2021-2022)')),
        color=alt.Color('Growth_Rate', scale=alt.Scale(scheme='greens')),
        tooltip=['Region', alt.Tooltip('Growth_Rate', format='.1%')]
    ).properties(title='Fig 9: Recovery Velocity - The "Revenge Travel" Surge (2021-2022)', height=350)

def chart_10_us_vs_thailand(df):
    """Fig 10: US vs Thailand"""
    target_countries = ['UNITED STATES OF AMERICA', 'THAILAND']
    df_comp = df[
        (df['Report Type'].str.contains('Inbound Tourism-Expenditure', na=False)) &
        (df['Metric'].isin(['Travel', 'Total'])) &
        (df['Country'].isin(target_countries)) &
        (df['Year'].between(2019, 2022))
    ].copy()
    
    df_comp['Index'] = 0.0
    for country in target_countries:
        try:
            base_val = df_comp[(df_comp['Country'] == country) & (df_comp['Year'] == 2019)]['Value'].values[0]
            df_comp.loc[df_comp['Country'] == country, 'Index'] = (df_comp['Value'] / base_val) * 100
        except IndexError: continue
        
    return alt.Chart(df_comp).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('Year:O'),
        y=alt.Y('Index', title='Recovery Index (2019=100)'),
        color='Country',
        tooltip=['Country', 'Year', alt.Tooltip('Index', format='.1f')]
    ).properties(title='Fig 10: Case Study - The Divergence (USA vs. Thailand)', height=350)

# --- 4. MAIN APP LOGIC ---

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("TSA Data Science")
        st.caption("Category: Data Science and Analytics")
        st.caption("Team: Viraat Chauhan & Pranav Sreepada")
        st.divider()
        app_mode = st.radio("Select Module:", ["Portfolio Report Visuals", "Interactive Data Explorer"])
        st.divider()
        st.info("**Project Title:**\nTHE GLOBAL STANDSTILL")
        st.caption("Data Source: UNWTO (2015-2022)")

    df = load_data()
    if df.empty: st.stop()
    df = add_regions(df)

    # --- HERO HEADER ---
    st.markdown("""
        <div class="header-container">
            <p class="category-tag">TSA Data Science & Analytics Portfolio</p>
            <h1 class="main-title">THE GLOBAL STANDSTILL</h1>
            <div class="meta-data-container">
                <div class="meta-item">
                    <span class="meta-label">Team</span>
                    <span class="meta-value">Viraat Chauhan & Pranav Sreepada</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Topic</span>
                    <span class="meta-value">Regional Analysis of COVID-19 Tourism Economics</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Data Source</span>
                    <span class="meta-value">UNWTO (2015-2022)</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- TAB 1 ---
    if app_mode == "Portfolio Report Visuals":
        st.subheader("Portfolio Visuals (Chapters 4-8)")
        st.markdown("These 10 visualizations correspond directly to the figures cited in the written portfolio report.")
        st.divider()
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("#### Baseline & Recovery")
            st.altair_chart(chart_1_baseline(df), use_container_width=True)
            st.caption("**Fig 1:** Regional Inbound Expenditure (2015-2022)")
            st.divider()
            st.altair_chart(chart_3_recovery(df), use_container_width=True)
            st.caption("**Fig 3:** The Two-Speed Recovery (Index 2019=100)")
            st.divider()
            st.markdown("#### Economic Impact")
            st.altair_chart(chart_5_infrastructure_emptying(df), use_container_width=True)
            st.caption("**Fig 5:** Infrastructure Emptying (Spain)")
            st.divider()
            st.altair_chart(chart_7_absolute_loss(df), use_container_width=True)
            st.caption("**Fig 7:** Absolute Economic Loss (2019-2020)")
            st.divider()
            st.markdown("#### Future Outlook")
            st.altair_chart(chart_9_recovery_velocity(df), use_container_width=True)
            st.caption("**Fig 9:** Recovery Velocity (2021-2022)")
            
        with col2:
            st.markdown("#### Shock & Vulnerability")
            st.altair_chart(chart_2_shock(df), use_container_width=True)
            st.caption("**Fig 2:** The Asymmetric Shock (% Drop)")
            st.divider()
            st.altair_chart(chart_4_transport_vulnerability(df), use_container_width=True)
            st.caption("**Fig 4:** Transport Vulnerability (Air vs Land)")
            st.divider()
            st.markdown("#### Financial Resilience")
            st.altair_chart(chart_6_balance_payments(df), use_container_width=True)
            st.caption("**Fig 6:** Balance of Payments Cushion")
            st.divider()
            st.altair_chart(chart_8_top10_drops(df), use_container_width=True)
            st.caption("**Fig 8:** Top 10 Hardest Hit Countries")
            st.divider()
            st.markdown("#### Case Study")
            st.altair_chart(chart_10_us_vs_thailand(df), use_container_width=True)
            st.caption("**Fig 10:** USA vs Thailand Divergence")

    # --- TAB 2 ---
    elif app_mode == "Interactive Data Explorer":
        st.subheader("Interactive Data Explorer")
        st.markdown("Use this tool to explore the raw UNWTO dataset.")
        
        with st.expander("📊 Filter Options", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                countries = sorted(df['Country'].unique())
                default_countries = ['SPAIN', 'THAILAND', 'UNITED STATES OF AMERICA', 'FRANCE']
                default_selection = [c for c in default_countries if c in countries]
                sel_country = st.multiselect("Select Countries", countries, default=default_selection)
            with col2:
                metrics = sorted(df['Metric'].unique())
                sel_metric = st.selectbox("Select Metric", metrics, index=0)
            with col3:
                reports = sorted(df['Report Type'].unique())
                sel_report = st.selectbox("Select Report Type", ["All"] + reports)
            min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
            sel_years = st.slider("Select Year Range", min_year, max_year, (2015, 2022))

        df_filtered = df[(df['Country'].isin(sel_country)) & (df['Year'].between(sel_years[0], sel_years[1]))]
        if sel_metric: df_filtered = df_filtered[df_filtered['Metric'] == sel_metric]
        if sel_report != "All": df_filtered = df_filtered[df_filtered['Report Type'] == sel_report]
            
        if not df_filtered.empty:
            # Stats & KPIs
            
            # --- KPI 1: Cumulative Volume (Meaningful Scale) ---
            total_vol = df_filtered['Value'].sum()
            
            # --- KPI 2: Peak Value (Best Performance) ---
            peak_val = df_filtered['Value'].max()
            
            # --- KPI 3: Trend (Simple Growth if applicable) ---
            # If we have at least 2 years, show latest growth
            df_yearly = df_filtered.groupby('Year')['Value'].sum().sort_index()
            if len(df_yearly) >= 2:
                latest_val = df_yearly.iloc[-1]
                prev_val = df_yearly.iloc[-2]
                trend_pct = (latest_val - prev_val) / prev_val
                trend_label = f"Trend ({df_yearly.index[-1]} vs {df_yearly.index[-2]})"
            else:
                trend_pct = 0
                trend_label = "Trend (Insufficient Data)"

            st.markdown("### Key Performance Indicators (KPIs)")
            kpi1, kpi2, kpi3 = st.columns(3)
            
            kpi1.metric("Cumulative Volume", f"{total_vol:,.0f}", help="Total sum of values for selected period")
            kpi2.metric("Peak Yearly Volume", f"{peak_val:,.0f}", help="Highest single data point found")
            kpi3.metric(trend_label, f"{trend_pct:+.1%}", delta_color="normal")
            
            # Interactive Chart with Smart Labels
            st.markdown("### Trend Visualization")
            
            # Smart Y-Axis Labeling
            y_title = "Value" # Default
            if "Expenditure" in sel_report:
                y_title = "Expenditure (Millions USD)" # Assuming millions based on dataset
            elif "Transport" in sel_report:
                y_title = "Arrivals (Count)"
            elif "Accommodation" in sel_report:
                y_title = "Nights (Count)"
            elif len(sel_metric) < 15:
                y_title = sel_metric
            
            chart = alt.Chart(df_filtered).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X('Year:O', title='Fiscal Year', axis=alt.Axis(labelAngle=0, grid=True)),
                # Fix: format inside Axis, Dynamic Title
                y=alt.Y('Value', title=y_title, axis=alt.Axis(format=',.0f'), scale=alt.Scale(zero=False)),
                color=alt.Color('Country', legend=alt.Legend(title="Market Region", orient='bottom')),
                tooltip=[
                    alt.Tooltip('Country', title='Market Region'),
                    alt.Tooltip('Year', title='Fiscal Period'),
                    alt.Tooltip('Value', title='Current Volume', format=',.0f'),
                    alt.Tooltip('Metric', title='Metric Type')
                ]
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
            
            # Data Table
            st.markdown("### Raw Data Table")
            pivot_table = df_filtered.pivot_table(index='Country', columns='Year', values='Value')
            st.dataframe(pivot_table)
            
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download CSV", data=csv, file_name='tsa_selected_data.csv', mime='text/csv')
        else: st.warning("No data found.")

if __name__ == "__main__":
    main()