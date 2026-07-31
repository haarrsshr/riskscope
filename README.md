# RiskScope 🔎

**RiskScope** is a free, zero-cost portfolio risk analytics dashboard inspired by institutional tools like BlackRock's Aladdin platform. It allows users to assess the historical risk and performance of a custom portfolio without needing a database, paid APIs, or server costs. 

This project demonstrates strong risk management principles and software engineering practices by building a robust risk analytics infrastructure entirely on open-source and free tools.

## Tech Stack
| Component | Technology |
|---|---|
| **UI & Framework** | Streamlit |
| **Market Data** | yfinance (Yahoo Finance) |
| **Data Manipulation** | Pandas, NumPy |
| **Risk Math** | SciPy |
| **Visualization** | Plotly |

## Key Formulas & Metrics
| Metric | Description |
|---|---|
| **Annualized Return** | Compound Annual Growth Rate (CAGR) extrapolated from historical cumulative return. |
| **Annualized Volatility** | `Daily Standard Deviation × √252` (assuming 252 trading days). |
| **Sharpe Ratio** | `(Annualized Return - Risk-Free Rate) / Annualized Volatility` |
| **Historical VaR** | Empirical percentile of historical daily returns. |
| **Parametric VaR** | Value-at-Risk assuming a normal distribution (`mean + z_score * std`). |
| **Max Drawdown** | Largest peak-to-trough decline of cumulative portfolio returns. |

## Local Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate  
   # On Windows:
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

## How to Use It
1. Open the dashboard in your browser (usually `http://localhost:8501`).
2. In the left sidebar, enter a comma-separated list of ticker symbols (e.g., `AAPL,MSFT,TLT,GLD`).
3. Enter the corresponding weights (must sum to 1.0, e.g., `0.3,0.3,0.2,0.2`).
4. Select a lookback period and define your benchmark ticker.
5. Set the current Risk-Free Rate (%).
6. Click **Analyze Portfolio**.

## Free Deployment (Streamlit Community Cloud)
You can deploy this dashboard for free:
1. Push your code to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**, select your repository, branch, and `app.py` as the main file path.
4. Click **Deploy**. Your app will be live with no credit card required!

## Known Limitations
* **VaR Assumptions:** Historical VaR assumes the past is a perfect predictor of the future. Parametric VaR assumes returns are normally distributed (which they are not in reality—they have "fat tails").
* **Correlation Instability:** Correlations between assets often break down and converge toward 1.0 during severe market crashes.
* **API Limits:** `yfinance` relies on Yahoo Finance, which may rate-limit you if you make too many requests in a short period.

## Possible Extensions
* Monte Carlo VaR simulations for better tail risk estimation.
* Factor exposure regression (e.g., Fama-French 3-factor model).
* Automated rebalancing suggestions based on target weights.
* Multi-portfolio comparison (Portfolio A vs Portfolio B).
