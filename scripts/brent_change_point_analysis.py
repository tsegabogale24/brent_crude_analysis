# Cell 3: Imports and functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import pymc as pm
import arviz as az

def load_and_preprocess(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError("CSV must contain 'Date' and 'Price' columns.")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
    df.dropna(subset=['Date'], inplace=True)

    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Price'] = df['Price'].interpolate(method='time')
    df.dropna(subset=['Price'], inplace=True)

    df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
    df['volatility_30d'] = df['log_return'].rolling(window=30).std()
    df.dropna(inplace=True)
    return df

def detect_change_points_ruptures(df: pd.DataFrame, n_bkps=3, model="rbf"):
    algo = rpt.Pelt(model=model).fit(df['Price'].values)
    bkps = algo.predict(pen=10)
    print(f"Detected change points at indices: {bkps}")

    rpt.display(df['Price'].values, bkps)
    plt.title('Change Point Detection (ruptures)')
    plt.show()

    change_dates = df.index[[b - 1 for b in bkps[:-1]]]
    print("Change point dates (ruptures):", list(change_dates))

    return change_dates

def bayesian_change_point_model(log_returns: np.ndarray, dates: pd.DatetimeIndex):
    n = len(log_returns)
    x = np.arange(n)

    with pm.Model() as model:
        tau = pm.DiscreteUniform('tau', lower=0, upper=n - 1)
        mu1 = pm.Normal('mu1', mu=0, sigma=0.05)
        mu2 = pm.Normal('mu2', mu=0, sigma=0.05)
        sigma = pm.HalfNormal('sigma', sigma=0.02)

        mu = pm.math.switch(tau >= x, mu1, mu2)
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=log_returns)

        trace = pm.sample(
            draws=1000,
            tune=500,
            chains=2,
            cores=1,
            return_inferencedata=True,
            progressbar=True
        )

    az.plot_trace(trace, var_names=["tau", "mu1", "mu2", "sigma"])
    plt.show()

    summary = az.summary(trace, var_names=["tau", "mu1", "mu2", "sigma"])
    print(summary)

    tau_samples = trace.posterior['tau'].values.flatten()
    most_probable_tau_idx = int(np.round(tau_samples.mean()))
    change_date = dates[most_probable_tau_idx]
    print(f"Most probable Bayesian change point date: {change_date.date()}")

    plt.hist(tau_samples, bins=n, alpha=0.7)
    plt.title('Posterior Distribution of Change Point (tau)')
    plt.xlabel('Time index')
    plt.ylabel('Frequency')
    plt.show()

    az.plot_posterior(trace, var_names=['mu1', 'mu2'])  # Removed credible_interval
    plt.show()

    mean_before = trace.posterior['mu1'].mean().item()
    mean_after = trace.posterior['mu2'].mean().item()
    pct_change = (np.exp(mean_after) - np.exp(mean_before)) / np.exp(mean_before) * 100

    print(f"Mean log return before change point: {mean_before:.5f}")
    print(f"Mean log return after change point:  {mean_after:.5f}")
    print(f"Estimated % change in average price after change point: {pct_change:.2f}%")

    return change_date, trace
