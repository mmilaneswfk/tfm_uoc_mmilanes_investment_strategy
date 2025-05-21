import pandas as pd
import numpy as np

def aggregate_strategies_for_plot(strategy_results, strategy_names, compute_cumulative=True, round_decimals=3):
    """
    Aggregates precomputed strategy series into a single DataFrame for plotting.

    Parameters:
    -----------
    strategy_results : list of pandas.Series
        List of time series returned by strategy functions, e.g.
        [strategy_1(valid_df), strategy_2(return_spy, target), ...].
    strategy_names : list of str
        Names to use for each series in the resulting DataFrame.
    compute_cumulative : bool, default=True
        Whether to convert each series to cumulative returns.
    round_decimals : int or None, default=3
        Decimal places to round the series (None skips rounding).

    Returns:
    --------
    pandas.DataFrame
        DataFrame with one column per strategy, indexed by date.
    """
    if len(strategy_results) != len(strategy_names):
        raise ValueError("strategy_results and strategy_names must have the same length")

    processed = {}
    for name, series in zip(strategy_names, strategy_results):
        s = series.copy()
        if compute_cumulative:
            s = s.cumsum()
        if round_decimals is not None:
            s = s.round(round_decimals)
        processed[name] = s

    return pd.DataFrame(processed)

def aggregate_lag_returns_for_plot(strategy_results, strategy_names, compute_cumulative=True, round_decimals=3):
    """
    Aggregates precomputed strategy returns for a specific lag period (default: 1) into a single DataFrame for plotting.

    Parameters:
    -----------
    strategy_results : list of pandas.DataFrame
        List of DataFrames with 'return_X' columns returned by strategy functions.
    strategy_names : list of str
        Names to use for each series in the resulting DataFrame.
    compute_cumulative : bool, default=True
        Whether to convert each series to cumulative returns.
    round_decimals : int or None, default=3
        Decimal places to round the series (None skips rounding).

    Returns:
    --------
    pandas.DataFrame
        DataFrame with one column per strategy, indexed by date.
    """
    if len(strategy_results) != len(strategy_names):
        raise ValueError("strategy_results and strategy_names must have the same length")

    processed = {}
    for name, result in zip(strategy_names, strategy_results):

        s = result.copy()
        if compute_cumulative:
            s = (1 + s).cumprod() - 1
        if round_decimals is not None:
            s = s.round(round_decimals)
        processed[name] = s

    return pd.DataFrame(processed)

# Example strategy functions

def buy_model_signals(total_gains, **kwargs):
    """Strategy that buys based on model signals."""
    return total_gains

def constant_investment_model(valid, target, **kwargs):
    """Strategy with constant investment based on model."""
    result = valid.abs().mul(target).dropna().apply(
        lambda x: x.sum()/np.count_nonzero(x) if np.count_nonzero(x) > 0 else 0, 
        axis=1
    ).fillna(0)
    return result

def buy_all_sectors(potential_gains, **kwargs):
    """Strategy that buys all sectors."""
    return potential_gains

def top_sector_with_model(top_sector_analysis, **kwargs):
    """Strategy that buys top sector based on model."""
    return top_sector_analysis['actual_return']

def sp500_benchmark(returns_spy, reference_index, **kwargs):
    """S&P 500 benchmark strategy."""
    return returns_spy.loc[reference_index]

def average_sector_strategy(target_df, **kwargs):
    """
    Strategy that equally weights investment across all sectors.
    
    Parameters:
    -----------
    target_df : pandas.DataFrame
        DataFrame with dates as index and sectors as columns containing returns.
    **kwargs : dict
        Additional keyword arguments (not used but included for consistency).
    
    Returns:
    --------
    pandas.Series
        Average returns across all sectors for each date.
    """
    # Calculate mean across all sectors for each date
    return target_df.mean(axis=1)


def top_n_periodic_strategy(predicted_proba, target, N, valid=None, **kwargs):
    """
    Selects the top N sectors for each date based on predicted probability
    and returns the average daily return of those sectors, considering validity.

    Parameters:
    -----------
    predicted_proba : pandas.DataFrame
        Date‐indexed probabilities for each sector.
    target : pandas.DataFrame
        Date‐indexed returns for each sector (same index & columns as predicted_proba).
    N : int
        Number of top‐probability sectors to pick for each date.
    valid : pandas.DataFrame or None
        Same shape as target/ predicted_proba, with 1 for valid invest days per sector.
        If None, all days are considered valid.
    **kwargs : dict
        Ignored (for API consistency).

    Returns:
    --------
    pandas.Series
        Strategy returns indexed by date.
    """
    # ensure same index on predictions and returns
    if not predicted_proba.index.equals(target.index) or not predicted_proba.columns.equals(target.columns):
        raise ValueError("predicted_proba and target must share the same index and columns")
    # handle valid mask
    if valid is None:
        valid = pd.DataFrame(1, index=target.index, columns=target.columns)
    else:
        if not valid.index.equals(target.index) or not valid.columns.equals(target.columns):
            raise ValueError("valid must share the same index and columns as target")
        
    # predicted_proba = predicted_proba.shift(1, freq = 'W-SUN').iloc[:-1]
    # valid = valid.shift(1, freq = 'W-SUN').iloc[:-1]

    daily_returns = []
    dates = predicted_proba.index

    for date in dates:
        # 1) Get probabilities for the current date
        proba_today = predicted_proba.loc[date]
        # 2) Identify top N sectors for this date
        top_sectors_today = proba_today.nlargest(N).index
        # 3) Get returns and validity for these sectors on this date
        returns_today = target.loc[date, top_sectors_today]
        valid_today = valid.loc[date, top_sectors_today]
        # 4) Filter returns based on validity
        valid_returns_today = returns_today[valid_today.abs() == 1]
        # 5) Calculate mean return for valid sectors, default to 0 if none are valid
        if not valid_returns_today.empty:
            mean_return = valid_returns_today.mean()
        else:
            mean_return = 0.0
        daily_returns.append(mean_return)

    # Create the result Series
    result_series = pd.Series(daily_returns, index=dates, name='top_n_periodic_returns')

    return result_series.sort_index()

