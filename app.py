import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

# Suppress yfinance warnings for cleaner output
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RiskLens", layout="wide", page_icon="🔎")

# --- Helper functions ---

@st.cache_data(ttl=3600)
def fetch_historical_data(tickers, benchmark, period_str):
    """
    Fetches historical adjusted close prices for given tickers and benchmark.
    """
    period_map = {'1y': '1y', '3y': '3y', '5y': '5y', '10y': '10y'}
    yf_period = period_map.get(period_str, '5y')
    
    all_tickers = tickers + [benchmark]
    
    # auto_adjust=True returns Adjusted Close as 'Close'
    data = yf.download(all_tickers, period=yf_period, auto_adjust=True, progress=False)
    
    if data.empty:
        return pd.DataFrame(), pd.Series()
        
    if 'Close' in data:
        closes = data['Close']
    else:
        closes = data
        
    if isinstance(closes, pd.Series):
        closes = closes.to_frame()
        if len(all_tickers) == 1:
            closes.columns = all_tickers
            
    available_tickers = [t for t in tickers if t in closes.columns]
    port_prices = closes[available_tickers] if available_tickers else pd.DataFrame()
    
    bench_prices = closes[benchmark] if benchmark in closes.columns else pd.Series(dtype=float)
    
    return port_prices, bench_prices

@st.cache_data(ttl=3600)
def fetch_scenario_data(tickers, start_date, end_date):
    """
    Fetches historical data for specific stress test date ranges.
    """
    # Buffer to ensure we can calculate returns on the first day
    start_buffer = pd.to_datetime(start_date) - timedelta(days=7)
    # yfinance end date is exclusive, so we add 1 day
    end_date_dt = pd.to_datetime(end_date) + timedelta(days=1)
    
    data = yf.download(tickers, start=start_buffer, end=end_date_dt, auto_adjust=True, progress=False)
    
    if data.empty:
        return pd.DataFrame()
        
    if 'Close' in data:
        closes = data['Close']
    else:
        closes = data
        
    if isinstance(closes, pd.Series):
        closes = closes.to_frame()
        if len(tickers) == 1:
            closes.columns = tickers
            
    return closes

# --- Risk Engine Functions ---

def compute_portfolio_returns(prices, weights):
    """Computes daily weighted portfolio returns."""
    daily_returns = prices.pct_change().dropna()
    port_returns = (daily_returns * weights).sum(axis=1)
    return port_returns

def annualized_volatility(daily_returns):
    """Calculates annualized volatility assuming 252 trading days."""
    return daily_returns.std() * np.sqrt(252)

def sharpe_ratio(annualized_return, risk_free_rate, ann_vol):
    """Calculates the Sharpe ratio."""
    if ann_vol == 0:
        return 0.0
    return (annualized_return - risk_free_rate) / ann_vol

def historical_var(daily_returns, confidence_level=0.95):
    """Calculates historical Value at Risk (VaR)."""
    return np.percentile(daily_returns, 100 * (1 - confidence_level))

def parametric_var(daily_returns, confidence_level=0.95):
    """Calculates parametric (normal) Value at Risk (VaR)."""
    mu = np.mean(daily_returns)
    sigma = np.std(daily_returns)
    return stats.norm.ppf(1 - confidence_level, mu, sigma)

def max_drawdown(daily_returns):
    """Calculates the maximum drawdown of the portfolio."""
    cum_returns = (1 + daily_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max
    return drawdowns.min()

def total_return(daily_returns):
    """Calculates total cumulative return."""
    return (1 + daily_returns).prod() - 1

def annualized_return(daily_returns):
    """Calculates annualized return."""
    total_ret = total_return(daily_returns)
    years = len(daily_returns) / 252
    if years <= 0:
        return 0.0
    return (1 + total_ret) ** (1 / years) - 1

# --- Main App ---

st.title("RiskLens 🔎")
st.subheader("Zero-Cost Portfolio Risk Analytics Dashboard")

# Sidebar
st.sidebar.header("Portfolio Parameters")
tickers_input = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,MSFT,TLT,GLD")
weights_input = st.sidebar.text_input("Weights (comma-separated)", "0.3,0.3,0.2,0.2")
lookback = st.sidebar.selectbox("Lookback Period", ["1y", "3y", "5y", "10y"], index=2)
benchmark = st.sidebar.text_input("Benchmark", "SPY")
rf_rate_input = st.sidebar.number_input("Risk-Free Rate (Annual %)", value=4.0)
rf_rate = rf_rate_input / 100.0

analyze_btn = st.sidebar.button("Analyze Portfolio")

if not analyze_btn:
    st.info("👈 Configure your portfolio in the sidebar and click **Analyze Portfolio** to view risk metrics.")
    st.markdown("### Example Portfolios to Try")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**60/40 Classic**")
        st.code("Tickers: SPY,BND\nWeights: 0.6,0.4")
    with col2:
        st.markdown("**All Weather**")
        st.code("Tickers: VTI,TLT,IEF,GLD,DBC\nWeights: 0.3,0.4,0.15,0.075,0.075")
    with col3:
        st.markdown("**Tech Heavy**")
        st.code("Tickers: AAPL,MSFT,NVDA,QQQ\nWeights: 0.25,0.25,0.2,0.3")
else:
    # --- Input Validation ---
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    try:
        weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
    except ValueError:
        st.error("Weights must be numeric values separated by commas.")
        st.stop()
        
    if len(tickers) != len(weights):
        st.error(f"Mismatch: {len(tickers)} tickers and {len(weights)} weights provided.")
        st.stop()
        
    if not np.isclose(sum(weights), 1.0, atol=0.01):
        st.error(f"Weights sum to {sum(weights):.4f}. They must sum to 1.0.")
        st.stop()
        
    benchmark = benchmark.strip().upper()
    if not benchmark:
        st.error("Please provide a benchmark ticker.")
        st.stop()
        
    # --- Data Fetching ---
    with st.spinner("Fetching historical market data..."):
        port_prices, bench_prices = fetch_historical_data(tickers, benchmark, lookback)
        
    if port_prices.empty or bench_prices.empty:
        st.error("Failed to fetch historical data. Please check your tickers and benchmark.")
        st.stop()
        
    # Check for missing data
    missing_tickers = [t for t in tickers if t not in port_prices.columns or port_prices[t].isna().all()]
    if missing_tickers:
        st.error(f"Could not fetch data for: {', '.join(missing_tickers)}. Please check the tickers.")
        st.stop()
        
    # Drop rows with any NaN to ensure we analyze the period where all assets existed
    port_prices = port_prices.dropna()
    bench_prices = bench_prices.dropna()
    
    # Align dates
    common_dates = port_prices.index.intersection(bench_prices.index)
    port_prices = port_prices.loc[common_dates]
    bench_prices = bench_prices.loc[common_dates]
    
    if len(port_prices) < 20:
        st.error("Not enough overlapping historical data for these tickers to perform meaningful analysis.")
        st.stop()
        
    # --- Calculations ---
    port_returns = compute_portfolio_returns(port_prices, weights)
    bench_returns = bench_prices.pct_change().dropna()
    
    ann_ret = annualized_return(port_returns)
    ann_vol = annualized_volatility(port_returns)
    sharpe = sharpe_ratio(ann_ret, rf_rate, ann_vol)
    mdd = max_drawdown(port_returns)
    var_95 = historical_var(port_returns, 0.95)
    var_99 = historical_var(port_returns, 0.99)
    
    # --- UI Dashboard ---
    st.markdown("## Portfolio Performance & Risk Summary")
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Ann. Return", f"{ann_ret*100:.2f}%")
    m2.metric("Ann. Volatility", f"{ann_vol*100:.2f}%")
    m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m4.metric("Max Drawdown", f"{mdd*100:.2f}%")
    m5.metric("1-Day VaR (95%)", f"{var_95*100:.2f}%")
    m6.metric("1-Day VaR (99%)", f"{var_99*100:.2f}%")
    
    # --- Cumulative Returns Chart ---
    st.markdown("### Cumulative Growth of $1")
    cum_port = (1 + port_returns).cumprod()
    cum_bench = (1 + bench_returns).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port, mode='lines', name='Portfolio', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, mode='lines', name=f'Benchmark ({benchmark})', line=dict(width=2, dash='dash', color='gray')))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Value ($)",
        template="plotly_white",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Correlation Heatmap ---
    st.markdown("### Asset Correlation (Daily Returns)")
    daily_asset_returns = port_prices.pct_change().dropna()
    corr_matrix = daily_asset_returns.corr()
    
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        color_continuous_scale='RdBu_r', 
        zmin=-1, zmax=1,
        aspect="auto"
    )
    fig_corr.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # --- Stress Testing Engine ---
    st.markdown("## Historical Stress Tests")
    st.write("Replaying historical market crashes against your *current* portfolio weights.")
    
    scenarios = {
        "2008 Global Financial Crisis": ("2008-09-01", "2009-03-31"),
        "2020 COVID Crash": ("2020-02-15", "2020-03-23"),
        "2022 Rate Hike Selloff": ("2022-01-01", "2022-10-31")
    }
    
    cols = st.columns(len(scenarios))
    
    for i, (scen_name, (start_date, end_date)) in enumerate(scenarios.items()):
        with cols[i]:
            st.markdown(f"**{scen_name}**\n\n*{start_date} to {end_date}*")
            try:
                scen_prices = fetch_scenario_data(tickers, start_date, end_date)
                
                if not scen_prices.empty:
                    scen_prices = scen_prices.dropna(how='all')
                    scen_prices = scen_prices.ffill()
                    
                    # Ensure we only use dates within the strict scenario window for returns
                    # We buffered the start date to calculate the first day's return
                    scen_prices = scen_prices.loc[:end_date]
                    
                    if len(scen_prices) > 1:
                        valid_tickers = [t for t in tickers if t in scen_prices.columns and not scen_prices[t].isna().all()]
                        
                        if len(valid_tickers) == len(tickers):
                            # Filter the exact dates of the stress test scenario for presentation
                            scen_returns_full = compute_portfolio_returns(scen_prices, weights)
                            scen_returns_window = scen_returns_full.loc[start_date:end_date]
                            
                            if len(scen_returns_window) > 0:
                                s_tot_ret = total_return(scen_returns_window)
                                s_worst_day = scen_returns_window.min()
                                s_mdd = max_drawdown(scen_returns_window)
                                
                                st.metric("Total Return", f"{s_tot_ret*100:.2f}%")
                                st.metric("Worst Single Day", f"{s_worst_day*100:.2f}%")
                                st.metric("Max Drawdown", f"{s_mdd*100:.2f}%")
                            else:
                                st.warning("Not enough trading days in this period.")
                        else:
                            missing = set(tickers) - set(valid_tickers)
                            st.warning(f"Incomplete data for: {', '.join(missing)}")
                    else:
                        st.warning("Not enough data for this period.")
                else:
                    st.warning("No data retrieved for this period.")
            except Exception as e:
                st.warning(f"Data unavailable ({str(e)})")
                
    # --- Educational Section ---
    st.markdown("---")
    with st.expander("ℹ️ How to interpret these numbers"):
        st.markdown("""
        * **Value at Risk (VaR):** A 95% 1-Day VaR of -2.5% means there is a 5% chance the portfolio will lose more than 2.5% in a single day. *Limitation:* VaR assumes normal market conditions and does not capture "tail risk" (extreme black swan events). This is why we use stress testing alongside it.
        * **Sharpe Ratio:** Measures risk-adjusted return. A ratio > 1 is generally considered good. It tells you how much excess return you are receiving for the extra volatility you endure for holding a riskier asset.
        * **Max Drawdown:** The largest peak-to-trough drop in portfolio value. It's a key indicator of historical tail risk.
        * **Correlation Breakdown:** During extreme market crashes (like 2008 or 2020), assets that normally move independently often see their correlations converge toward 1. This means diversification can disappear exactly when you need it most.
        """)
