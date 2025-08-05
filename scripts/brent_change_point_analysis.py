import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import pymc as pm
import arviz as az


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    Loads Brent oil price data from CSV, handles mixed date formats, computes log returns and volatility.
    """
    df = pd.read_csv(filepath)

    # Clean column names and validate
    df.columns = df.columns.str.strip()
    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError("CSV must contain 'Date' and 'Price' columns.")

    # Robust date parsing (handles multiple formats)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
    df.dropna(subset=['Date'], inplace=True)

    # Sort and set index
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    # Ensure Price is numeric
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Price'] = df['Price'].interpolate(method='time')
    df.dropna(subset=['Price'], inplace=True)

    # Calculate log returns and volatility
    df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
    df['volatility_30d'] = df['log_return'].rolling(window=30).std()

    df.dropna(inplace=True)  # drop rows with NA from shift/rolling
    return df


def detect_change_points_ruptures(df: pd.DataFrame, n_bkps=3, model="rbf"):
    """
    Detects structural breakpoints in price using the Ruptures library.
    """
    algo = rpt.Pelt(model=model).fit(df['Price'].values)
    bkps = algo.predict(pen=10)
    print(f"Detected change points at indices: {bkps}")

    # Plot
    rpt.display(df['Price'].values, bkps)
    plt.title('Change Point Detection on Brent Oil Prices (ruptures)')
    plt.show()

    # Extract change point dates
    change_dates = df.index[[b - 1 for b in bkps[:-1]]]  # drop final index (len)
    print("Change point dates (ruptures):", list(change_dates))

    return change_dates


def bayesian_change_point_model(log_returns: np.ndarray, dates: pd.DatetimeIndex):
    """
    Bayesian change point detection using PyMC3 on log return series.
    Optimized for faster sampling by reducing draws and chains.
    """
    n = len(log_returns)
    x = np.arange(n)

    with pm.Model() as model:
        tau = pm.DiscreteUniform('tau', lower=0, upper=n - 1)
        mu1 = pm.Normal('mu1', mu=0, sigma=0.05)
        mu2 = pm.Normal('mu2', mu=0, sigma=0.05)
        sigma = pm.HalfNormal('sigma', sigma=0.02)

        mu = pm.math.switch(tau >= x, mu1, mu2)
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=log_returns)

        # Reduced samples and tuning steps, fewer chains and cores
        trace = pm.sample(
            draws=1000, 
            tune=500, 
            chains=2, 
            cores=1, 
            return_inferencedata=True,
            progressbar=True
        )

    # Diagnostics and visuals
    az.plot_trace(trace, var_names=["tau", "mu1", "mu2", "sigma"])
    plt.show()

    summary = az.summary(trace, var_names=["tau", "mu1", "mu2", "sigma"])
    print(summary)

    tau_samples = trace.posterior['tau'].values.flatten()
    most_probable_tau_idx = int(np.round(tau_samples.mean()))
    change_date = dates[most_probable_tau_idx]
    print(f"Most probable Bayesian change point date: {change_date.date()}")

    # Posterior tau
    plt.hist(tau_samples, bins=n, alpha=0.7)
    plt.title('Posterior Distribution of Change Point (tau)')
    plt.xlabel('Time index')
    plt.ylabel('Frequency')
    plt.show()

    # Posterior mean returns
    az.plot_posterior(trace, var_names=['mu1', 'mu2'], credible_interval=0.95)
    plt.show()

    # Quantitative impact
    mean_before = trace.posterior['mu1'].mean().item()
    mean_after = trace.posterior['mu2'].mean().item()
    pct_change = (np.exp(mean_after) - np.exp(mean_before)) / np.exp(mean_before) * 100

    print(f"Mean log return before change point: {mean_before:.5f}")
    print(f"Mean log return after change point:  {mean_after:.5f}")
    print(f"Estimated % change in average price after change point: {pct_change:.2f}%")

    return change_date, trace


def save_data(df: pd.DataFrame, path: str):
    df.to_csv(path)
    print(f" Data saved to {path}")


# Main analysis pipeline
if __name__ == "__main__":
    data_path = "../data/raw/BrentOilPrices.csv"
    save_path = "../data/processed/brent_oil_with_log_and_volatility.csv"

    print(" Loading and preprocessing data...")
    df = load_and_preprocess(data_path)

    print("\n Detecting change points with ruptures...")
    ruptures_cps = detect_change_points_ruptures(df, n_bkps=3)

    print("\n Running Bayesian Change Point Model with PyMC3...")
    change_date, trace = bayesian_change_point_model(df['log_return'].values, df.index)

    print("\n Saving enriched data...")
    save_data(df, save_path)

    print("\n Analysis Complete.")
