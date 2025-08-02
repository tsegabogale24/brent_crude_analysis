from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats


def load_brent_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    return df

def plot_time_series(df: pd.DataFrame, column: str = 'Price') -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df[column], label='Brent Crude Price')
    plt.title('Brent Crude Oil Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rolling_mean(df: pd.DataFrame, column: str = 'Price', window: int = 30) -> None:
    rolling = df[column].rolling(window=window).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(df[column], label='Original')
    plt.plot(rolling, label=f'{window}-Day Moving Average', color='red')
    plt.title('Trend Detection in Brent Crude Prices')
    plt.legend()
    plt.grid(True)
    plt.show()

def seasonal_decompose_plot(df: pd.DataFrame, column: str = 'Price', period: int = 365) -> None:
    result = seasonal_decompose(df[column], model='multiplicative', period=period)
    result.plot()
    plt.show()

def adf_test(df: pd.DataFrame, column: str = 'Price') -> None:
    result = adfuller(df[column].dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')
    if result[1] < 0.05:
        print("✅ The series is stationary (reject H0).")
    else:
        print("❌ The series is non-stationary (fail to reject H0).")

def compute_log_returns(df: pd.DataFrame, price_col: str = 'Price') -> pd.DataFrame:
    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    return df

def plot_log_returns(df: pd.DataFrame, column: str = 'log_return') -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df[column], label='Log Returns')
    plt.title('Log Returns of Brent Crude Prices')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_volatility(df: pd.DataFrame, return_col: str = 'log_return', window: int = 20) -> pd.DataFrame:
    df['volatility'] = df[return_col].rolling(window=window).std()
    return df

def plot_volatility(df: pd.DataFrame, column: str = 'volatility') -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df[column], label='Volatility (Rolling Std Dev)')
    plt.title('Volatility Clustering in Brent Crude Prices')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_return_distribution(df: pd.DataFrame, column: str = 'log_return') -> None:
    """
    Plots the histogram of log returns with KDE.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column].dropna(), bins=50, kde=True)
    plt.title("Distribution of Log Returns")
    plt.xlabel("Log Return")
    plt.grid(True)
    plt.show()

def plot_autocorrelation(df: pd.DataFrame, column: str = 'log_return', lags: int = 40):
    """
    Plots the ACF and PACF of the log returns.
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(df[column].dropna(), ax=ax[0], lags=lags)
    plot_pacf(df[column].dropna(), ax=ax[1], lags=lags)
    ax[0].set_title("Autocorrelation of Log Returns")
    ax[1].set_title("Partial Autocorrelation of Log Returns")
    plt.tight_layout()
    plt.show()
   
