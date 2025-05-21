#%% Import libraries and data
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from datetime import datetime
import numpy as np
import yaml
from statsmodels.regression.rolling import RollingOLS
import os
import warnings
import time
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# Load configuration from YAML file
# Determine the path to the configuration file
config_path = os.path.join(os.path.dirname(__file__), '../configs/create_dataset_config.yml')

# Load configuration from YAML file
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    # Load time constants from configuration
    time_constants = config.get('time_constants', {})
    WEEKS_PER_YEAR = time_constants.get('WEEKS_PER_YEAR', 52)
    WEEKS_PER_MONTH = time_constants.get('WEEKS_PER_MONTH', 4)
    YEARS_FOR_TICKER_VALIDITY = time_constants.get('YEARS_FOR_TICKER_VALIDITY', 5)
    YEARS_FOR_ROLLING_WINDOW = time_constants.get('YEARS_FOR_ROLLING_WINDOW', 2)
    MIN_PERIODS_FOR_ROLLING = time_constants.get('MIN_PERIODS_FOR_ROLLING', 6)
    
    # Calculate derived constants
    MIN_ENTRIES = WEEKS_PER_YEAR * YEARS_FOR_TICKER_VALIDITY
    ROLLING_WINDOW_SIZE = WEEKS_PER_YEAR * YEARS_FOR_ROLLING_WINDOW
    ROLLING_BETA_WINDOW = WEEKS_PER_MONTH * YEARS_FOR_ROLLING_WINDOW * 12  # 2 years in weeks
    
    # Load other configuration items
    DATA_STORE = config['DATA_STORE']
    START = config['start_year']
    INTERVAL = config['interval']
    ticker_list = config['ticker_list']
    lag_list = config['lags']
    normalize = config['normalize']
    neutralize = config['neutralize']
    outlier_cutoff = config['outlier_cutoff']
    excess_return = config['excess_return']
    momentum_list = config['momentum']
    lagged_returns = config['lagged_returns']
    fred_indicators = config['fred_indicators']
    fred_features = config['fred_features']
    fred_label = config['fred_label']
    fama_french_factors = config['fama_french_factors']
    columns_to_drop = config['columns_to_drop']
    fred_shifts = config['fred_shifts']
    CHECK_NULLS_DATA = config['check_nulls_data']
    chunk_size = config['chunk_size']
    chunk_delay_seconds = config['chunk_delay_seconds']
    DOWNLOAD_YFINANCE = config['DOWNLOAD_YFINANCE']
    YFINANCE_TEMP = config['YFINANCE_TEMP']


# Download data using yfinance
# Check if we should download data or read from parquet
if DOWNLOAD_YFINANCE:
    data = None

    # Process tickers in chunks
    for i in range(0, len(ticker_list), chunk_size):
        chunk_tickers = ticker_list[i:i+chunk_size]
        
        print(f"Downloading data for tickers {i+1}-{min(i+chunk_size, len(ticker_list))} of {len(ticker_list)}")
        
        chunk_data = yf.download(
            tickers=chunk_tickers,
            group_by='ticker',
            auto_adjust=False,
            interval=INTERVAL,
            start=START,
            threads=False
        )
        
        # Handle single ticker case
        if len(chunk_tickers) == 1 and not isinstance(chunk_data.columns, pd.MultiIndex):
            ticker = chunk_tickers[0]
            chunk_data.columns = pd.MultiIndex.from_product([[ticker], chunk_data.columns])
        
        # Combine with previous data
        if data is None:
            data = chunk_data
        else:
            data = pd.concat([data, chunk_data], axis=1)
        
        # Wait before the next chunk (if not the last one)
        if i + chunk_size < len(ticker_list) and chunk_delay_seconds > 0:
            print(f"Waiting {chunk_delay_seconds} seconds before next chunk...")
            time.sleep(chunk_delay_seconds)
    
    # Save data to parquet
    parquet_path = os.path.join(os.path.dirname(__file__), YFINANCE_TEMP)
    data.to_parquet(parquet_path)
    print(f"Data saved to {parquet_path}")
else:
    # Try to read from parquet file
    parquet_path = os.path.join(os.path.dirname(__file__), YFINANCE_TEMP)
    if os.path.exists(parquet_path):
        data = pd.read_parquet(parquet_path)
        print(f"Data loaded from {parquet_path}")
    else:
        raise FileNotFoundError(f"Parquet file {parquet_path} not found. Set DOWNLOAD_YFINANCE to True to download data.")

#%%
# Check for missing values in the last 5 weeks of close data
if CHECK_NULLS_DATA:
    # Get the last 5 weeks of data
    last_n_weeks = 5
    last_weeks_data = data.iloc[-last_n_weeks:] if len(data) >= last_n_weeks else data
    
    # Check for null values in Close columns
    missing_tickers = []
    for ticker in ticker_list:
        if (ticker, 'Close') in data.columns and last_weeks_data[(ticker, 'Close')].isnull().any():
            missing_tickers.append(ticker)
        
    if missing_tickers:
        raise ValueError(f"Missing close values found in the last {last_n_weeks} weeks for tickers: {', '.join(missing_tickers)}")


data.index = data.index.tz_localize('UTC')

data = (data
    .stack(-2, future_stack=True)
    .rename_axis(['date', 'ticker'])
)
data = data.loc[str(START):]

# Assert all tickers are present in the data
missing_tickers = [ticker for ticker in ticker_list if ticker not in data.index.get_level_values('ticker').unique()]
assert not missing_tickers, f"Missing tickers: {missing_tickers}"

# Assert there is data for every year expected
start_year, end_year = int(START), datetime.now().year
years = range(start_year, end_year + 1)

for ticker in ticker_list:
    ticker_data = data.xs(ticker, level='ticker')
    missing_years = [year for year in years if year not in ticker_data.index.year.unique()]
    assert not missing_years, f"Ticker {ticker} is missing data for years: {missing_years}"

new_order = ['Open', 'High', 'Low','Close','Volume', 'Adj Close']
new_names = ['open', 'high', 'low','close','volume', 'adj_close']

data = data[new_order]
data.columns = new_names

# Extract SPY data and remove duplicates
spy_prices = data.xs('SPY', level='ticker')[['close']]
spy_prices = spy_prices[~spy_prices.index.duplicated(keep='first')]

# Get prices without SPY, remove duplicates, and reorganize by ticker
prices = data.drop('SPY', level='ticker')[['close']]
prices = prices[~prices.index.duplicated(keep='first')].unstack('ticker')

# Convert to weekly data
prices = prices.resample('W').last()
spy_prices = spy_prices.resample('W').last()

print("Configuration and ETF prices successfully read.")

#%% Create a new dataframe for returns
returns = pd.DataFrame()


for lag in lag_list:
    returns[f'return_{lag}'] = (
        (prices
         .pct_change(periods=lag)
         .stack(future_stack=True)
         .groupby(level='ticker').transform(lambda x: x.clip(lower=float(x.quantile(outlier_cutoff)),
                       upper=float(x.quantile(1-outlier_cutoff))))
         + 1)
        .pow(1/lag)
        .sub(1)
    )

# Drop rows with NaN values that result from the lag calculation
returns = returns.swaplevel().dropna()

# Drop tickers with less than 5 years of data
valid_tickers = returns.index.get_level_values('ticker').value_counts()[lambda x: x >= MIN_ENTRIES].index

# Filter the returns dataframe to keep only valid tickers
returns = returns.loc[returns.index.get_level_values('ticker').isin(valid_tickers)]
returns_raw = returns.copy()

# Define a function to normalize by rolling standard deviation with a specified window
def normalize_by_rolling_std(series, window=ROLLING_WINDOW_SIZE, min_periods=MIN_PERIODS_FOR_ROLLING):
    """
    Normalize a series by its rolling standard deviation.

    Parameters:
    series (pd.Series): The input series to normalize.
    window (int): The window size for the rolling standard deviation. Default is 52 weeks (1 year).
    min_periods (int): Minimum number of observations in the window required to have a value. Default is 6.

    Returns:
    pd.Series: The normalized series.
    """
    return (series / series.rolling(window, closed='left', min_periods=min_periods).std())

# Define a function to neutralize the group
def neutralize(group):
    return (group - group.mean()) / group.std()

if excess_return:
    # Create a new dataframe for SPY returns
    returns_spy = pd.DataFrame()

    for lag in lag_list:
        returns_spy[f'return_{lag}'] = (
            (spy_prices
                .pct_change(periods=lag)
                .transform(lambda x: x.clip(lower=float(x.quantile(outlier_cutoff)),
                            upper=float(x.quantile(1-outlier_cutoff))))
                + 1)
            .pow(1/lag)
            .sub(1)
        )
  

    # Drop rows with NaN values that result from the lag calculation
    returns_spy = returns_spy.dropna()
    returns_spy_raw = returns_spy.copy()

    returns = returns.sub(returns_spy)

# Normalize the returns if the normalize flag is set in the configuration
if normalize:
    normalized_returns = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        normalized_returns[col] = (
            returns[col]
            .groupby(level='ticker')
            .transform(lambda x: normalize_by_rolling_std(x))
        )
    returns = normalized_returns.dropna()

# Neutralize the returns if the neutralize flag is set in the configuration
if neutralize:
    neutralized_returns = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        neutralized_returns[col] = (
            returns[col]
            .groupby(level='date')
            .transform(neutralize)
        )
    returns = neutralized_returns

# if excess_return:
#     # Create a new dataframe for SPY returns
#     returns_spy = pd.DataFrame()

#     for lag in lag_list:
#         returns_spy[f'return_{lag}'] = (
#             (spy_prices
#                 .pct_change(periods=lag)
#                 .transform(lambda x: x.clip(lower=float(x.quantile(outlier_cutoff)),
#                             upper=float(x.quantile(1-outlier_cutoff))))
#                 + 1)
#             .pow(1/lag)
#             .sub(1)
#         )
  

#     # Drop rows with NaN values that result from the lag calculation
#     returns_spy = returns_spy.dropna()
#     returns_spy_raw = returns_spy.copy()

#     # Normalize the SPY returns if the normalize flag is set in the configuration
#     if normalize:
#         normalized_spy_returns = pd.DataFrame(index=returns_spy.index)
#         for col in returns_spy.columns:
#             normalized_spy_returns[col] = normalize_by_rolling_std(returns_spy[col])
#         returns_spy = normalized_spy_returns.dropna()

#     # # Neutralize the SPY returns if the neutralize flag is set in the configuration
#     # if neutralize:
#     #     returns_spy = neutralize(returns_spy)

#     returns = returns.sub(returns_spy)

returns_spy.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '')),
                                        '../output/returns_spy.csv'), index=True)

# Check if returns dataframe is not empty and has the right schema and format
if not returns.empty and all(col.startswith('return_') for col in returns.columns):
    print("Returns dataframe has been created successfully with the correct schema and format.")
else:
    print("Returns dataframe is either empty or does not have the correct schema and format.")

#%% Fama-French factors
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 
                             'famafrench', start=START)[0].drop('RF', axis=1)
factor_data.index = factor_data.index.to_timestamp().tz_localize('UTC')
factor_data = factor_data.resample('W').last().div(100) #weekly
factor_data.index.name = 'date'
factor_data = factor_data.join(returns['return_1']).sort_index()

# Calculate betas as you currently do
T = ROLLING_BETA_WINDOW  # 2 years in weeks
betas = (factor_data.groupby(level='ticker',
                             group_keys=False)
         .apply(lambda x: RollingOLS(endog=x.return_1,
                                     exog=sm.add_constant(x.drop('return_1', axis=1)),
                                     window=min(T, x.shape[0]-1))
                .fit(params_only=True)
                .params
                .drop('const', axis=1)))

# Create a properly time-aligned betas dataframe using merge_asof
betas_lagged = pd.DataFrame()

# Process each ticker to maintain the alignment
for ticker in betas.index.get_level_values('ticker').unique():
    # Get betas for this ticker
    ticker_betas = betas.xs(ticker, level='ticker').reset_index()
    
    # Get dates from returns for this ticker
    ticker_returns = returns.xs(ticker, level='ticker').reset_index()
    
    # Use merge_asof to find the most recent beta for each return date
    # Similar to how you handle FRED data
    aligned = pd.merge_asof(
        ticker_returns[['date']],  # Only keep the date column
        ticker_betas,
        on='date',
        direction='backward'
    )
    
    # Set ticker and recreate MultiIndex
    aligned['ticker'] = ticker
    aligned.set_index(['ticker', 'date'], inplace=True)
    
    # Add to our result
    betas_lagged = pd.concat([betas_lagged, aligned])

# Impute missing values as in your original code
betas_lagged[fama_french_factors] = betas_lagged[fama_french_factors].groupby(
    level='ticker', group_keys=False
).apply(lambda x: x.fillna(x.mean()))

# Join with returns to create data_joined
data_joined = returns.join(betas_lagged)

# for lag in momentum_list: #for weeks
#     data_joined[f'momentum_{lag}'] = data_joined[f'return_{lag}'].sub(data_joined.return_1)
# data_joined[f'momentum_3_12'] = data_joined[f'return_12'].sub(data_joined.return_3)

dates = data_joined.index.get_level_values('date')
data_joined['month'] = dates.month

# Fill NA values in the 'sector' column using the ticker information
data_joined['sector'] = data_joined.index.get_level_values('ticker')

# Lagged returns
for t in range(lagged_returns['start'], lagged_returns['end']):
    data_joined[f'return_1_t-{t}'] = data_joined.groupby(level='ticker',group_keys=False).return_1.shift(t)

for t in lag_list:
    data_joined[f'target_{t}'] = data_joined.groupby(level='ticker')[f'return_{t}'].shift(-t)

# Check if Fama-French factors are present in data_joined and print a message
if any(factor in data_joined.columns for factor in fama_french_factors):
    print("Fama-French features were aggregated successfully.")

#%% Fred data
# data_fred = (web.DataReader(fred_indicators, 'fred', start_year, end_year+1)
#         .ffill()
#         .resample('W')
#         .last()
#         # .dropna()
#         )
# data_fred.columns = fred_label + fred_features

# # Aplicar el shift a cada columna según su offset
# for col, offset in fred_shifts.items():
#     if col in data_fred.columns:
#         # El shift con freq desplaza el índice, simulando la fecha en que el dato estuvo disponible.
#         # Luego reindexamos para que la serie quede alineada al índice original del DataFrame.
#         data_fred[col] = data_fred[col].shift(1, freq=offset).reindex(data_fred.index)

data_fred = web.DataReader(fred_indicators, 'fred', start_year, end_year+1)
data_fred.columns = fred_label + fred_features

# Apply shift to columns according to their offset
for col, offset in fred_shifts.items():
    if col in data_fred.columns:
        # El shift con freq desplaza el índice, simulando la fecha en que el dato estuvo disponible.
        # Luego reindexamos para que la serie quede alineada al índice original del DataFrame.
        series_temp = data_fred[col].shift(1, freq=offset)
        data_fred[col] = (pd.merge_asof(data_fred.drop(columns = col), 
                                        series_temp.groupby(series_temp.index).last(), 
                                        left_index=True, right_index=True, 
                                        direction='backward')
                          [col])

data_fred = data_fred.ffill().resample('W').last()[fred_label + fred_features]

for column in data_fred.columns:
    data_fred[column + '_diff'] = data_fred[column].diff().replace(0, np.nan).ffill()
    data_fred[column + '_chg'] = data_fred[column].pct_change().replace(0, np.nan).ffill()

# Load columns to drop from the configuration
existing_columns_to_drop = [col for col in columns_to_drop if col in data_fred.columns]
data_fred = data_fred.drop(existing_columns_to_drop, axis=1)
data_fred.index = data_fred.index.tz_localize('UTC')

data_fred.index.name = 'date'

# Final join
data_final = (data_joined
        .join(data_fred))

# Check if Fred data is present in data_final and print a message
if any(indicator in data_final.columns for indicator in fred_indicators):
    print("Fred data has been successfully retrieved and added to the final dataset.")

#%% Saves data to HDF5 file
with pd.HDFStore(os.path.join(os.path.dirname(__file__), DATA_STORE)) as store:
    store.put('engineered_features', data_final.sort_index(), format='table', data_columns=True)
    store.put('data_raw', returns_raw.sort_index(), format='table', data_columns=True)  # before normalization of returns
    store.put('spy_raw', returns_spy_raw.sort_index(), format='table', data_columns=True)  # before normalization of returns

# Check if data has been successfully stored
with pd.HDFStore(os.path.join(os.path.dirname(__file__), DATA_STORE)) as store:
    if 'engineered_features' in store and 'data_raw' in store:
        print("Data has been successfully stored in the HDF5 file.")
    else:
        print("Data storage in the HDF5 file failed.")

print("Script finished.")

