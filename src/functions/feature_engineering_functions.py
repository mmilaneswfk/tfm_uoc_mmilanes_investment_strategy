import pandas as pd
import numpy as np
from sklego.preprocessing import RepeatingBasisFunction

def create_lags(df: pd.DataFrame, column: str, lags: list, prefix: str = '') -> pd.DataFrame:
    """
    Create lagged features for a given column, respecting group boundaries at level 0
    
    Args:
        df: DataFrame with MultiIndex (level 0: group, level 1: timestamp)
        column: Column name to create lags for
        lags: List of lag periods to create
        prefix: Optional prefix for the lag column names (default: '')
    
    Returns:
        DataFrame with added lag columns
    """
    result = df.copy()
    
    # Create lags within each group
    for lag in lags:
        result[f'{prefix}lag_{lag}'] = (
            df.groupby(level=0)[column]
            .shift(lag)
        )
    
    return result

def create_rolling_features(df: pd.DataFrame, column: str, rolling_params: list, prefix: str = '') -> pd.DataFrame:
    """
    Create rolling features for a given column, respecting group boundaries at level 0
    
    Args:
        df: DataFrame with MultiIndex (level 0: group, level 1: timestamp)
        column: Column name to create rolling features for
        rolling_params: List of tuples (window, lag, agg_func)
        prefix: Optional prefix for the rolling column names (default: '')
    
    Returns:
        DataFrame with added rolling features
    """
    result = df.copy()
    
    for window, lag, agg_func in rolling_params:
        feature_name = f'{prefix}roll_{window}_{lag}_{agg_func}'
        result[feature_name] = (
            result.groupby(level=0)[column]
            .rolling(window=window, min_periods=1)
            .agg(agg_func)
            .droplevel(0)
        )

        result[feature_name] = (
            result.groupby(level=0)[feature_name]
            .shift(lag)
        )
    
    return result

def create_momentum_features(df: pd.DataFrame, momentum_params: list) -> pd.DataFrame:
    """
    Create momentum features by subtracting returns and applying lag
    
    Args:
        df: DataFrame with MultiIndex (level 0: group, level 1: timestamp)
        momentum_params: List of tuples (a1, a2, lag) to create momentum features
    
    Returns:
        DataFrame with added momentum features
    """
    result = df.copy()
    
    for a1, a2, lag in momentum_params:
        feature_name = f'momentum_{a1}_{a2}_{lag}'
        momentum = result[f'return_{a1}'] - result[f'return_{a2}']
        result[feature_name] = momentum
        result[feature_name] = (
            result.groupby(level=0)[feature_name]
            .shift(lag)
        )
    
    return result

def create_time_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Create time-based features from the timestamp in level 1 of the index
    
    Args:
        df: DataFrame with MultiIndex (level 0: group, level 1: timestamp)
        features: List of time features to create. Valid options: 
                 ['dayofyear', 'weekofyear', 'month', 'quarter', 'dayofweek', 
                  'is_month_start', 'is_month_end']
    
    Returns:
        DataFrame with added time features
    """
    result = df.copy()
    timestamp = pd.DatetimeIndex(result.index.get_level_values(1))
    
    feature_mappings = {
        'dayofyear': lambda x: x.dayofyear,
        'weekofyear': lambda x: x.isocalendar().week,
        'month': lambda x: x.month,
        'quarter': lambda x: x.quarter,
        'dayofweek': lambda x: x.dayofweek,
        'is_month_start': lambda x: x.is_month_start.astype(int),
        'is_month_end': lambda x: x.is_month_end.astype(int)
    }
    
    for feature in features:
        if feature in feature_mappings:
            result[feature] = feature_mappings[feature](timestamp).astype(int).values
    
    return result

def create_time_encoding_rbf(df: pd.DataFrame, periods: dict, n_columns: int) -> pd.DataFrame:
    """
    Create cyclic time features using RepeatingBasisFunction
    
    Args:
        df: DataFrame with MultiIndex (level 0: group, level 1: timestamp)
        periods: Dictionary with time attribute as key (e.g. 'dayofyear') and list of periods as value
        n_columns: Number of output columns to generate for each period
    
    Returns:
        DataFrame with added cyclic time features
    """
    result = df.copy()
    
    feature_mappings = {
        'dayofyear': lambda x: x.dayofyear,
        'weekofyear': lambda x: x.isocalendar().week,
        'month': lambda x: x.month,
        'quarter': lambda x: x.quarter,
        'dayofweek': lambda x: x.dayofweek,
        'is_month_start': lambda x: x.is_month_start.astype(int),
        'is_month_end': lambda x: x.is_month_end.astype(int)
    }
    
    for time_attr, period in periods.items():
        # Get time attribute
        timestamp = pd.DatetimeIndex(df.index.get_level_values(1))
        time_values = feature_mappings[time_attr](timestamp)
        temp_df = pd.DataFrame({time_attr: time_values})
        
        # Create RBF features 
        rbf = RepeatingBasisFunction(
            n_periods=n_columns,
            column=time_attr,
            input_range=(1, period),
            remainder="drop"
        )
        rbf.fit(temp_df)
        rbf_features = rbf.transform(temp_df)
        
        # Add features to result DataFrame
        for i in range(rbf_features.shape[1]):
            result[f'time_rbf_{time_attr}_{period}_{i+1}'] = rbf_features[:, i]
    
    return result

def create_diff(df: pd.DataFrame, diff_params: list) -> pd.DataFrame:
    """
    Create difference features for given columns and periods, respecting group boundaries at level 0
    
    Args:
        df: DataFrame with MultiIndex (level 0: group, level 1: timestamp)
        diff_params: List of [column, periods] pairs to create differences for
    
    Returns:
        DataFrame with added difference columns
    """
    result = df.copy()
    for column, periods in diff_params:
        result[f'{column}_diff_{periods}'] = (
            df.groupby(level=0)[column]
            .diff(periods=periods)
        )
    return result

def simple_labelling(series: pd.Series, threshold: float = 0) -> pd.Series:
    """
    Create binary labels based on whether values exceed a threshold
    
    Args:
        series: Input series to be labeled
        threshold: Value above which label will be 1 (default: 0)
    
    Returns:
        Series with binary labels (1 if value > threshold, 0 otherwise, NaN if input is NaN)
    """
    # Create a result series initialized with NaN values
    result = pd.Series(np.nan, index=series.index)
    
    # Apply the comparison only to non-null values
    mask = series.notnull()
    result[mask] = (series[mask] > threshold).astype(int)
    
    return result

def last_target_outcomes(df: pd.DataFrame, target_name: str, threshold: float = 0) -> pd.DataFrame:
    """
    Creates features based on past target outcomes
    
    Args:
        df: DataFrame with MultiIndex (level 0: group, level 1: timestamp)
        target_name: Name of the target column
        threshold: Threshold for binary labeling
    
    Returns:
        DataFrame with added target outcome features
    """
    result = df.copy()
    
    # Create binary labels
    result['target_binary'] = simple_labelling(result[target_name], threshold)
    
    # Create lag of binary target
    result = create_lags(result, 'target_binary', [1,2,4,8,12,26,52], prefix = 'fe_target_last_')
    
    # Create rolling features
    rolling_params = [
        (12, 1, 'mean'),
        (12, 1, 'sum'),
        (4, 1, 'mean'),
        (4, 1, 'sum')
    ]
    result = create_rolling_features(result, 'target_binary', rolling_params, prefix = 'fe_target_last_')
    # Drop the temporary target_binary column
    result = result.drop('target_binary', axis=1)
    
    return result

def create_cyclical_features(df: pd.DataFrame, time_features: list) -> pd.DataFrame:
    """
    Create sine and cosine features for cyclical time features
    
    Args:
        df: DataFrame with MultiIndex (level 0: group, level 1: timestamp)
        time_features: List of time features to create cyclical features for.
                        Valid options: ['dayofyear', 'weekofyear', 'month', 'quarter', 'dayofweek']
    
    Returns:
        DataFrame with added sine and cosine features
    """
    result = df.copy()
    timestamp = pd.DatetimeIndex(result.index.get_level_values(1))
    
    # Define max values for each cycle
    max_values = {
        'dayofyear': 366,  # Account for leap years
        'weekofyear': 53,  # ISO calendar can have 53 weeks
        'month': 12,
        'quarter': 4,
        'dayofweek': 7
    }
    
    # Define functions to extract each feature
    feature_extractors = {
        'dayofyear': lambda x: x.dayofyear,
        'weekofyear': lambda x: x.isocalendar().week,
        'month': lambda x: x.month,
        'quarter': lambda x: x.quarter,
        'dayofweek': lambda x: x.dayofweek
    }
    
    for feature in time_features:
        if feature in max_values:
            # Extract the time feature
            values = feature_extractors[feature](timestamp)
            max_value = max_values[feature]
            
            # Calculate sine and cosine features
            result.loc[:, f'{feature}_sin'] = pd.Series(np.sin(2 * np.pi * values / max_value)).astype(float).values
            result.loc[:, f'{feature}_cos'] = pd.Series(np.cos(2 * np.pi * values / max_value)).astype(float).values
    
    return result