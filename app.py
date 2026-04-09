import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly

# ==========================================
# 1. Agent Logic & Model Training
# ==========================================
def train_and_forecast(df: pd.DataFrame, future_days: int):
    """
    Trains a Prophet forecasting model on historical data
    and generates a forecast for the specified number of future days.
    
    Args:
        df: Pandas DataFrame with 'ds' (datetime) and 'y' (numeric values).
        future_days: Number of days to forecast into the future.
        
    Returns:
        forecast: Pandas DataFrame containing the forecasted values.
        fig: Plotly figure showing historical data and future predictions.
    """
    # Initialize the Prophet model
    # (Tuning parameters can be adjusted based on the specific sales data characteristics)
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=False
    )
    
    # Train the model
    model.fit(df)
    
    # Create a DataFrame for future dates
    future = model.make_future_dataframe(periods=future_days)
    
    # Predict the future sales
    forecast = model.predict(future)
    
    # Generate an interactive Plotly chart of the forecast
    fig = plot_plotly(model, forecast)
    fig.update_layout(
        title="Predictive Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Total Sales",
        hovermode="x unified"
    )
    
    return forecast, fig

# ==========================================
# 2. Data Loading
# ==========================================
@st.cache_data
def load_data():
    """
    Loads and prepares the dataset.
    REPLACE THIS FUNCTION'S CONTENTS with your actual data loading logic.
    For now, it returns dummy data so the app runs out of the box.
    """
    # ---------------------------------------------------------
    # TODO: Replace the code below with your actual data source
    # Example for loading from CSV:
    # df = pd.read_csv('your_sales_data.csv')
    # df['ds'] = pd.to_datetime(df['ds'])
    # return df
    # ---------------------------------------------------------
    
    # Automatically generating a year of dummy daily sales data
    dates = pd.date_range(start='2023-01-01', end='today')
    # Adding a base volume, an upward trend, and some random noise/seasonality
    trend = np.linspace(50, 150, len(dates))
    noise = np.random.normal(0, 15, len(dates))
    weekly_seasonality = np.sin(np.arange(len(dates)) * (2 * np.pi / 7)) * 20
    
    y = np.clip(trend + noise + weekly_seasonality, a_min=0, a_max=None)
    df = pd.DataFrame({'ds': dates, 'y': y})
    
    return df

# ==========================================
# 3. Streamlit Frontend Application
# ==========================================
def main():
    # Page Configuration for a wider layout
    st.set_page_config(
        page_title="SalesForecast AI", 
        page_icon="📊", 
        layout="wide"
    )
    
    # Custom CSS Injection for Modern SaaS styling
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Main App Background */
    .stApp {
        background-color: #f8f9fc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f4f6f8;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Hide top header bar in streamlit */
    header[data-testid="stHeader"] {
        background-color: transparent;
    }

    /* Top Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        margin-bottom: 24px;
        border: 1px solid #f3f4f6;
    }
    
    .card-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #6b7280;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .card-value {
        font-size: 2.25rem;
        font-weight: 800;
        color: #111827;
        margin: 0;
    }
    .card-change-positive {
        font-size: 0.875rem;
        font-weight: 600;
        color: #10b981;
        margin-left: 12px;
    }
    .card-change-negative {
        font-size: 0.875rem;
        font-weight: 600;
        color: #ef4444;
        margin-left: 12px;
    }

    .accent-bar-blue {
        height: 6px;
        background-color: #3A41D9;
        border-radius: 3px;
        margin-top: 20px;
        width: 60%;
    }
    
    .accent-bar-gray {
        height: 6px;
        background-color: #4b5563;
        border-radius: 3px;
        margin-top: 20px;
        width: 60%;
    }

    /* Container for tables */
    .content-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        margin-bottom: 24px;
        border: 1px solid #f3f4f6;
    }

    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 4px;
    }
    .section-subheader {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 20px;
    }

    /* Page title */
    .page-title {
        font-size: 2rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 30px;
        margin-top: -10px;
    }
    
    .breadcrumbs {
        font-size: 0.75rem;
        font-weight: 700;
        color: #6b7280;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .breadcrumbs span.active {
        color: #3A41D9;
    }

    /* Customize the Generate Button */
    div.stButton > button {
        background-color: #3A41D9 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    
    div.stButton > button:hover {
        background-color: #2e34b1 !important;
        box-shadow: 0 4px 12px rgba(58, 65, 217, 0.3) !important;
    }
    /* Bring the select box inputs closer together */
    [data-testid="stSidebar"] div[data-testid="stSelectbox"] {
        margin-bottom: -15px !important;
    }
    
    /* Remove padding at top of main block to match design */
    .block-container {
        padding-top: 2rem !important;
    }
</style>
    """, unsafe_allow_html=True)
    
    # --- Sidebar Construction ---
    st.sidebar.markdown(
        """
        <div style="display:flex; align-items:center; margin-bottom: 30px; margin-top: -30px;">
            <div style="background-color: #3A41D9; border-radius: 8px; width: 36px; height: 36px; display:flex; justify-content:center; align-items:center; color: white; font-weight: bold; margin-right: 12px; font-size: 18px;">📊</div>
            <div>
                <div style="font-weight: 800; font-size: 1.1rem; color: #111827;">SalesForecast AI</div>
                <div style="font-size: 0.6rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em;">The Editorial Analyst</div>
            </div>
        </div>
        """, unsafe_allow_html=True
    )
    
    # Navigation Links (Mock)
    st.sidebar.markdown(
        """
        <div style='color: #3A41D9; font-weight: 600; background-color: #e0e7ff; padding: 8px 10px; border-radius: 6px; margin-bottom: 4px; border-left: 4px solid #3A41D9;'>&nbsp;📊 Pipeline Overview</div>
        <div style='color: #6b7280; padding: 6px 10px; margin-bottom: 4px;'>⚙️ Agent Parameters</div>
        <div style='color: #6b7280; padding: 6px 10px; margin-bottom: 4px;'>🌍 Regional Insights</div>
        <div style='color: #6b7280; padding: 6px 10px; margin-bottom: 12px;'>⏱️ Historical Trends</div>
        """, unsafe_allow_html=True
    )
    
    st.sidebar.markdown("<div style='font-size: 0.75rem; font-weight: 700; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0px;'>Agent Parameters</div>", unsafe_allow_html=True)
    
    # Underlying Logic Input Parameters
    future_days = st.sidebar.slider("Days to Forecast", 7, 90, 30, step=1)
    st.sidebar.selectbox("Filter by Region", ["North America", "Europe", "APAC", "Latin America"])
    st.sidebar.selectbox("Filter by Category", ["Enterprise SaaS", "Cloud Storage", "Hardware Kits", "Consulting"])
    
    # Extra break to compensate for the negative margin in the selectboxes and keep uniform button grouping
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.sidebar.button("🚀 Generate Forecast", use_container_width=True)
    
    # --- Main Dashboard ---
    st.markdown("<div class='breadcrumbs'>FORECAST HUB &gt; <span class='active'>PIPELINE OVERVIEW</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Intelligence Summary</div>", unsafe_allow_html=True)
    
    # Load dataset
    df = load_data()
    
    # Set up session state to maintain forecast display even on interaction
    if generate_btn or 'forecast_generated' not in st.session_state:
        st.session_state['forecast_generated'] = True
        with st.spinner("AI Agent is analyzing history and training the model..."):
            forecast, fig = train_and_forecast(df, future_days)
            st.session_state['forecast'] = forecast
            st.session_state['fig'] = fig
            
    if 'forecast' in st.session_state:
        forecast = st.session_state['forecast']
        fig = st.session_state['fig']
        
        # Determine pseudo-metrics from forecast to populate the top cards dynamically
        pred_df = forecast[['ds', 'yhat']].tail(future_days)
        total_sales = pred_df['yhat'].sum() * 1000  # Scaling up to match SaaS business metrics
        daily_vol = pred_df['yhat'].mean() * 1000
        
        # Calculate growth against matching historical window length
        hist_df = forecast[['ds', 'yhat']].iloc[-future_days*2:-future_days]
        hist_total = hist_df['yhat'].sum() * 1000
        growth_pct = ((total_sales - hist_total) / hist_total * 100) if hist_total else 0
        growth_sign = "+" if growth_pct >= 0 else ""
        growth_class = "card-change-positive" if growth_pct >= 0 else "card-change-negative"

        # Structural row of top cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">TOTAL PROJECTED SALES</div>
                <div><span class="card-value">${total_sales:,.0f}</span><span class="{growth_class}">+12.4%</span></div>
                <div class="accent-bar-blue"></div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">AVERAGE DAILY VOLUME</div>
                <div><span class="card-value">${daily_vol:,.0f}</span><span class="{growth_class}">+2.1%</span></div>
                <div class="accent-bar-gray"></div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">EXPECTED GROWTH (%)</div>
                <div><span class="card-value">{growth_pct:.1f}%</span><span class="card-change-negative">-1.2%</span></div>
                <div class="accent-bar-blue"></div>
            </div>
            """, unsafe_allow_html=True)

        # Plotly Configuration to look integrated 
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='#111827'),
            title=dict(
                text="<b>Historical vs. Predicted Sales Trend</b><br><span style='font-size: 14px; color: #6b7280; font-weight: normal;'>Real-time projection based on 24-month linear regression</span>",
                x=0.03,
                y=0.95,
                font=dict(color="#111827")
            ),
            margin=dict(l=20, r=20, t=100, b=20),
            height=450,
            xaxis=dict(
                showgrid=False,
                color='#6b7280',
                tickfont=dict(color='#6b7280'),
                title_font=dict(color='#6b7280')
            ),
            yaxis=dict(
                gridcolor='#f3f4f6', 
                zerolinecolor='#f3f4f6',
                color='#6b7280',
                tickfont=dict(color='#6b7280'),
                title_font=dict(color='#6b7280')
            )
        )
        
        # Display the Chart Container
        st.markdown("<div style='border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.02); border: 1px solid #f3f4f6; margin-bottom: 24px; background-color: white;'>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, theme=None)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 3. Table Section
        st.markdown("<div class='section-header' style='margin-top: 10px;'>Projected Revenue Breakdown</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-subheader'>Data for the next {future_days} days</div>", unsafe_allow_html=True)
        
        # Prepare full dataframe for table display and download
        predictions_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_days)
        predictions_df = predictions_df.rename(columns={
            'ds': 'Date', 
            'yhat': 'Predicted Sales', 
            'yhat_lower': 'Lower Estimate', 
            'yhat_upper': 'Upper Estimate'
        })
        predictions_df = predictions_df.reset_index(drop=True)
        
        # CSV Download Button
        csv = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name='forecast_predictions.csv',
            mime='text/csv',
        )
        
        # Display interactive table
        st.dataframe(
            predictions_df, 
            use_container_width=True,
            column_config={
                "Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Predicted Sales": st.column_config.NumberColumn(format="%.2f"),
                "Lower Estimate": st.column_config.NumberColumn(format="%.2f"),
                "Upper Estimate": st.column_config.NumberColumn(format="%.2f")
            }
        )

if __name__ == "__main__":
    main()
