#%% Imports and config
import warnings
warnings.filterwarnings('ignore')

import os
import datetime
import copy
import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from src.utils import MultipleTimeSeriesCV
from src.functions.backtesting_functions import (
    create_lgbm_dataset,
    save_valid_dfs,
    save_model,
    save_feature_importances,
    save_model_parameters,
    save_prediction_accuracy,
    save_categorical_info,
    save_gains_analysis,
    save_top_sector_analysis,
    save_correlation_data,
    save_prediction_proba_df,
    extract_predictions_proba,
    analyze_top_sector_predictions,
    compare_predictions,
    analyze_predictions_accuracy
)
from src.functions.feature_engineering_functions import simple_labelling
from sklearn.metrics import accuracy_score

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '')),
                           '../configs/backtesting_config.yml')

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    DATA_STORE = config['DATA_STORE']
    use_best_params = config['use_best_params']
    incremental_training = config['incremental_training']
    backtest_start_date = config['backtest_start_date']
    backtest_end_date = config['backtest_end_date']
    backtest_to_date = config['backtest_to_date']
    backtest_frequency = config['backtest_frequency']
    COVID_FILTER = config['COVID_FILTER']
    retrain_model = config['retrain_model']
    YEAR = config['YEAR']
    TARGET_THRESHOLD = config['TARGET_THRESHOLD']
    YEARS_TRAIN = config['YEARS_TRAIN']
    WEEKS_TEST = config['WEEKS_TEST']
    model_params = config['model_params']
    TARGET_NAME = config['TARGET_NAME']
    CV_SPLITS = config['CV_SPLITS']
    CV_PURGE = config['CV_PURGE']
    CATEGORICAL_FEATURES = config['CATEGORICAL_FEATURES']
    USE_RETURN_AS_WEIGHT = config['USE_RETURN_AS_WEIGHT']


DATA_STORE = os.path.join(os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '')), 
                          DATA_STORE)

with pd.HDFStore(DATA_STORE) as store:
    best_params = (store['best_params']
            .sort_index()).to_dict()
    best_params = {k: int(v) if isinstance(v, (int, float)) and v.is_integer() else v for k, v in best_params.items()}
    
with pd.HDFStore(DATA_STORE) as store:
    data = (store['modeling_data']
            .sort_index())
    
if not CATEGORICAL_FEATURES:
    CATEGORICAL_FEATURES = "auto"

# Determine the folder where this script is located
current_folder = os.path.dirname(os.path.abspath(__file__))
# Determine the parent folder of this script’s directory
parent_folder = os.path.abspath(os.path.join(current_folder, os.pardir))

y = data[TARGET_NAME]
X = data.drop(TARGET_NAME, axis=1)

if USE_RETURN_AS_WEIGHT:
    sample_weights = pd.read_csv(os.path.join(parent_folder, "output/weights.csv"),
                                parse_dates=True, index_col=0).abs()
    sample_weights = sample_weights.stack()
    sample_weights.index = sample_weights.index.swaplevel(1,0)
    sample_weights.index.names = X.index.names
else:
    sample_weights = None


# Define COVID period dates (ensure consistency with modeling script, maybe load from config)
COVID_START_DATE = pd.Timestamp('2020-02-20', tz='UTC')
COVID_END_DATE = pd.Timestamp('2020-06-30', tz='UTC')

# Apply COVID filter if enabled in config
if COVID_FILTER:
    print(f"Filtering out COVID period: {COVID_START_DATE.date()} to {COVID_END_DATE.date()}")
    
    # Create mask based on X's date index
    covid_mask = ~((X.index.get_level_values('date') >= COVID_START_DATE) & 
                   (X.index.get_level_values('date') <= COVID_END_DATE))
    
    initial_rows = len(X)
    
    # Apply filter to X and y
    X = X.loc[covid_mask]
    y = y.loc[covid_mask]
    
    # Apply filter to sample_weights if they exist
    if sample_weights is not None:
        # Ensure sample_weights index matches X before filtering
        if not sample_weights.index.equals(X.index):
             # Reindex weights to match the filtered X index (handles potential mismatches)
             sample_weights = sample_weights.reindex(X.index)
             # Note: Reindexing might introduce NaNs if weights were missing for some X rows. Handle as needed.
        else:
             # If indices already match (after filtering X/y), filter weights directly
             sample_weights = sample_weights.loc[covid_mask]


    removed_rows = initial_rows - len(X)
    print(f"Removed {removed_rows} rows due to COVID filter.")
else:
    print("COVID filter is disabled.")


#%%
# Load and apply feature selection if available
try:
    with pd.HDFStore(DATA_STORE) as store:
        selected_features = store['selected_features'].values
        # Apply feature selection *after* potential COVID filtering
        X = X[selected_features] 
except KeyError:
    print("No feature selection found in store, using all features")
except Exception as e:
    print(f"Warning: Error loading selected features: {str(e)}")
    print("Proceeding with all features")

# Process categorical features - convert to integer codes and set as category type
if CATEGORICAL_FEATURES != "auto":
    # Check for missing categorical features upfront
    missing_cats = [cat for cat in CATEGORICAL_FEATURES if cat not in data.columns]
    if missing_cats:
        raise ValueError(f"Categorical features not found in the data: {missing_cats}")
    
    # Process all categorical features at once
    for cat in CATEGORICAL_FEATURES:
        # Factorize converts to integer codes (0,1,2...) and set category dtype
        X[cat] = pd.factorize(X[cat], sort=True)[0]
        X[cat] = X[cat].astype('category')


# Labeling
y = simple_labelling(y, TARGET_THRESHOLD)

# Print summary of null values in y
total_nulls = y.isna().sum()
print(f"Total null values in y: {total_nulls}")
null_by_date = y.isna().groupby(level=1).sum()
print("Null values by date:")
print(null_by_date.sort_values(ascending=False).head())

# Determine last date and infer frequency
date_index = X.index.get_level_values('date')
last_date = date_index.max()
inferred_freq = date_index.freqstr or pd.infer_freq(date_index) or backtest_frequency

# Compare the period of last_date to the current period
last_period    = last_date.to_period(inferred_freq)
current_period = pd.Timestamp.now(tz='UTC').to_period(inferred_freq)

# Check if last_date is in current period and has nulls
nulls_last = int(null_by_date.get(last_date, 0))
if last_period == current_period and nulls_last > 0:
    print(f"Removing data for current period {current_period} due to {nulls_last} null target(s)")
    # Drop rows where the date level equals last_date
    mask = date_index != last_date
    X = X[mask]
    y = y[mask]

if use_best_params:
    model_params.update(best_params)
    model = lgb.LGBMClassifier(**model_params)
else:
    model = lgb.LGBMClassifier(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=15,
        **model_params
    )

# cv = MultipleTimeSeriesCV(
#         n_splits=CV_SPLITS*CV_PURGE,
#         train_period_length=YEAR * YEARS_TRAIN,
#         test_period_length=WEEKS_TEST,
#         lookahead=1,
#         date_idx='date',
#         shuffle=False)

if backtest_to_date:
    backtest_end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    backtest_date_values = pd.to_datetime(X.index.get_level_values('date').unique())
    if pd.to_datetime(backtest_end_date) not in backtest_date_values:
        backtest_end_date = backtest_date_values.max().strftime('%Y-%m-%d')



#%% Data Preprocessing

valid_results = []
valid_df_saved = []

backtest_dates = pd.date_range(start=backtest_start_date, 
                              end=backtest_end_date, 
                              freq=backtest_frequency)

model_pre = None
train_df_pre = None
valid_df_pre = None

for test_date in backtest_dates:
    train_df, valid_df = create_lgbm_dataset(X, y,
                                            lookback_period=YEARS_TRAIN * 52,
                                            n_test_periods=WEEKS_TEST,
                                            end_date=test_date,
                                            sample_weights=sample_weights,
                                            categorical_features=CATEGORICAL_FEATURES)

    if train_df is None or valid_df is None:
        print(f"Skipping {test_date}: Not enough data for train/valid split.")
        continue

    if retrain_model or len(valid_results) == 0:
        # Full retrain
        model.fit(train_df.data, train_df.label,
                 eval_set=[(valid_df.data, valid_df.label)],
                 categorical_feature=CATEGORICAL_FEATURES)
    elif incremental_training:
        # Incremental training on new data only
        model.fit(train_df.data, train_df.label,
                 init_model=model,
                 categorical_feature=CATEGORICAL_FEATURES)

    # Make predictions
    valid_pred = model.predict(valid_df.data)
    valid_pred_proba = model.predict_proba(valid_df.data)[:, 1]

    if model_pre is not None:
        # Calibrate the model
        calibrator = CalibratedClassifierCV(model_pre,
                                            method='sigmoid',
                                            cv='prefit') # Use prefit since model is already trained

        # Fit calibrator on the same training data used for the model in this iteration
        calibrator.fit(train_df_pre.data, train_df_pre.label)

        # Make calibrated predictions and probabilities
        calibrator_pred = calibrator.predict(valid_df.data)
        calibrator_pred_proba = calibrator.predict_proba(valid_df.data)[:, 1] # Get probability for the positive class
    else:
        # If no previous model, use the current model's predictions
        calibrator_pred = valid_pred
        calibrator_pred_proba = valid_pred_proba

    # Store previous model and data if needed for analysis or comparison

    model_pre = copy.deepcopy(model)
    train_df_pre = copy.deepcopy(train_df)
    valid_df_pre = copy.deepcopy(valid_df)
    # Store results
    if len(np.unique(valid_df.label)) == 1:
        # Handle cases where only one class is present in the validation set
        result = {
            'date': test_date,
            'predictions': valid_pred,
            'predictions_proba': valid_pred_proba,
            'true_labels': valid_df.label.values,
            'calibrator_predictions': calibrator_pred,
            'calibrator_predictions_proba': calibrator_pred_proba, # Added calibrated probabilities
            'validation_auc': np.nan, # Set to None when only one class is present
            'validation_ap': np.nan  # Set to None when only one class is present
        }
    else:
        # Standard case with multiple classes
        result = {
            'date': test_date,
            'predictions': valid_pred,
            'predictions_proba': valid_pred_proba,
            'true_labels': valid_df.label.values,
            'calibrator_predictions': calibrator_pred,
            'calibrator_predictions_proba': calibrator_pred_proba, # Added calibrated probabilities
            'validation_auc': roc_auc_score(valid_df.label, valid_pred_proba), # Use probabilities for AUC
            'validation_ap': average_precision_score(valid_df.label, valid_pred_proba) # Use probabilities for AP
        }
    valid_results.append(result)
    valid_df_saved.append(valid_df)

# Concatenate valid_df_saved datasets
concatenated_valid_df = lgb.Dataset(
    np.vstack([df.data for df in valid_df_saved]),
    label=np.concatenate([df.label for df in valid_df_saved]),
    categorical_feature=CATEGORICAL_FEATURES
)

# %% Post-Backtesting Analysis

# Load returns data needed for subsequent analyses
returns_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '')),
                                      "../output/target.csv"), index_col=0, parse_dates=True)
returns_df.index = pd.to_datetime(returns_df.index) # Ensure index is datetime

# Get unique sectors from the data index
sectors = sorted(X.index.get_level_values('ticker').unique())

# Analyze prediction accuracy for both original and calibrated predictions
prediction_accuracy_df = analyze_predictions_accuracy(valid_results, sectors)
prediction_accuracy_df_calibrated = analyze_predictions_accuracy(valid_results, sectors, calibrated=True)

# Localize the index timezone to UTC for consistency
prediction_accuracy_df.index = prediction_accuracy_df.index.tz_localize('UTC')
prediction_accuracy_df_calibrated.index = prediction_accuracy_df_calibrated.index.tz_localize('UTC')

# Extract prediction probabilities for each sector
prediction_proba_df = extract_predictions_proba(valid_results, sectors)
calibrator_prediction_proba_df = extract_predictions_proba(valid_results, sectors, calibrator=True)

# Compare original and calibrated predictions
prediction_differences = compare_predictions(valid_results)

# Analyze the performance of selecting the top predicted sector
top_sector_analysis = analyze_top_sector_predictions(valid_results,
                                                     sectors,
                                                     returns_df)

# Calculate total gains based on prediction accuracy and actual returns
# Note: Assumes prediction_accuracy_df represents some form of signal strength or confidence
total_gains = prediction_accuracy_df.abs().mul(returns_df).dropna().sum(axis=1)

# Calculate potential gains (sum of returns across all sectors for the same period)
potential_gains = returns_df.loc[total_gains.index].sum(axis = 1)


#%% Correlation Analysis

last_3_years_date = X.index.get_level_values('date').max() - pd.DateOffset(years=3)
X_last_3_years = X[X.index.get_level_values('date') >= last_3_years_date]
y_last_3_years = y.loc[X_last_3_years.index]

returns_spy = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '')),
                                       "../output/returns_spy.csv"), parse_dates=True, index_col=0)['return_1'].sort_index().shift(-1)
spy_returns_filtered = returns_spy.loc[returns_spy.index.get_level_values('date') >= last_3_years_date]

combined_series = pd.Series(
    data=spy_returns_filtered.reindex(y.index.get_level_values('date')).values,
    index=y.index
)

spearman_corr = {}
for col in X_last_3_years.columns:
    corr, _ = spearmanr(X_last_3_years[col], combined_series.loc[X_last_3_years.index])
    spearman_corr[col] = corr

spearman_corr_spy = pd.Series(spearman_corr).sort_values(ascending=False)


# Compute Spearman correlation for each feature in X_last_3_years against y_last_3_years
sector_corrs = {}
sector_corrs['SPY'] = spearman_corr_spy

for sector in X_last_3_years.index.get_level_values(0).unique():
    sector_data = X_last_3_years.loc[sector]
    sector_y = y_last_3_years.loc[sector]
    
    # Calculate Spearman correlation
    spearman_corr = {}
    for col in sector_data.columns:
        corr, _ = spearmanr(sector_data[col], sector_y)
        spearman_corr[col] = corr

    spearman_corr_series = pd.Series(spearman_corr).sort_values(ascending=False)
    sector_corrs[sector] = spearman_corr_series


# Convert sector_corrs into a DataFrame with sectors as rows
correlation_data = pd.DataFrame(sector_corrs)


# Save the correlation_data dataframe
correlation_data_path = save_correlation_data(correlation_data, config_path)

# %% Saving results
# Call the function to save valid DataFrames
valid_df_path = save_valid_dfs(valid_df_saved, X, config_path)

top_sector_path = save_top_sector_analysis(top_sector_analysis, config_path)
gains_path = save_gains_analysis(total_gains, potential_gains, config_path)

# Execute saving functions
model_path = save_model(model, config_path)
importance_path = save_feature_importances(model, X, config_path)
params_path = save_model_parameters(model, config_path)
accuracy_path, calibrated_accuracy_path = save_prediction_accuracy(prediction_accuracy_df, prediction_accuracy_df_calibrated, config_path)
cat_info_path = save_categorical_info(CATEGORICAL_FEATURES if CATEGORICAL_FEATURES != "auto" else [], valid_df_path)

# call the saver
prediction_proba_path = save_prediction_proba_df(prediction_proba_df, config_path)
calibrator_prediction_proba_path = save_prediction_proba_df(calibrator_prediction_proba_df, config_path, calibrator=True)