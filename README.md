#  Brent Crude Oil Price Analysis - Task 1: Time Series Exploration

##  Business Objective

The objective of this project is to analyze how major global events impact Brent crude oil prices over time. Specifically, this task focuses on understanding the underlying time series properties of Brent prices (1987–2022), identifying significant trends, seasonality, stationarity, and volatility patterns. This groundwork is crucial for detecting structural breaks using change point analysis in subsequent tasks.

---

##  Task 1: Analyze Time Series Properties

###  Goals:
- Load and inspect historical Brent oil price data.
- Identify time series characteristics: trend, seasonality, stationarity.
- Perform statistical tests (ADF) and decompositions.
- Compute log returns and rolling volatility.
- Prepare data for change point detection.

---

## 📁 Folder Structure

```bash
brent-oil-analysis/
│
├── data/
│   ├── brent_oil.csv                  # Raw data from Yahoo Finance or EIA
│   └── brent_oil_with_log_and_volatility.csv  # Cleaned and enhanced dataset
    └── event_data.csv 
│
├── scripts/
│   └── brent_crude_analysis.py        # Modular functions for trend, seasonality, ADF test, log return, volatility
│
├── notebooks/
│   └──  brent_crude_analysis.ipynb       # Exploratory notebook using the functions
│   └──  load_event_data.ipynb
├
│   └── event_data.csv                 # Manually created CSV with major global events and dates
│
└── README.md                          # This file
 Methods & Implementation
📌 Step 1: Data Loading
Data is loaded from CSV and sorted by date.

📌 Step 2: Time Series Analysis
We explore:

Trend using rolling mean plots.

Seasonality using seasonal decomposition (statsmodels).

Stationarity using the Augmented Dickey-Fuller (ADF) test.

ADF Statistic: -1.99
p-value: 0.28
Conclusion: Non-stationary (fail to reject H0)
 Step 3: Log Returns
Log returns represent daily relative changes and help stabilize variance.

log_return 𝑡 = log(𝑃𝑡𝑃𝑡−1)
log_return t =log( P t−1 P t )
 Step 4: Volatility Clustering
Volatility is estimated using rolling standard deviation of log returns:

df["volatility_30d"] = df["log_return"].rolling(window=30).std()
This captures "volatility clustering", a common pattern in financial time series.
 Outputs
brent_oil_with_log_and_volatility.csv: contains date, price, log returns, and rolling volatility.

Time series plots for trend, decomposition, and stationarity checks.

Summary of ADF test result.
