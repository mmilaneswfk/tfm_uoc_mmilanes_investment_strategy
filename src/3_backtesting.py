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
from src.functions.optimization_functions import (
    parse_hyperparameter_space, average_precision_eval_sklearn, 
    average_precision_eval, purge_cv_folds, auc_feval
    )
from src.functions.backtesting_functions import (
    analyze_predictions_accuracy,
    analyze_top_sector_predictions,
    compare_predictions,
    create_lgbm_dataset,
    extract_predictions_proba,
    extract_predictions,
    extract_true_labels,
    save_categorical_info,
    save_dataframe_to_csv,  # New merged function
    save_model,
    save_valid_dfs,
    create_refined_hyperparameter_space
)
from src.functions.feature_engineering_functions import simple_labeling, labeling_selector, std_labeling
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import xgboost as xgb

# Add after other imports
import random
RANDOM_SEED = 42  # Set random seed for reproducibility

# Set random seeds for all libraries
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

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
    WEEKS_VALIDATION = config['WEEKS_VALIDATION']
    WEEKS_TEST = config['WEEKS_TEST']
    model_params = config['model_params']
    TARGET_NAME = config['TARGET_NAME']
    CV_SPLITS = config['CV_SPLITS']
    CV_PURGE = config['CV_PURGE']
    CATEGORICAL_FEATURES = config['CATEGORICAL_FEATURES']
    USE_RETURN_AS_WEIGHT = config['USE_RETURN_AS_WEIGHT']
    ENABLE_OPTIMIZATION = config['ENABLE_OPTIMIZATION']

config_hyper_path = os.path.join(os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '')),
                           '../configs/hyperparameters.yml')

with open(config_hyper_path, 'r') as file:
    config_hyper = yaml.safe_load(file)
    HYPERPARAMETER_SPACE = config_hyper['HYPERPARAMETER_SPACE']

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
COVID_END_DATE = pd.Timestamp('2020-07-30', tz='UTC')

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
        sample_weights = sample_weights.loc[X.index]

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
#y = simple_labeling(y, TARGET_THRESHOLD)
y = labeling_selector(y, 'simple')

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

# Check if last date has nulls and remove if necessary
nulls_last = int(null_by_date.get(last_date, 0))
if nulls_last > 0:
    print(f"Removing data for last date {last_date.date()} due to {nulls_last} null target(s)")
    # Drop rows where the date level equals last_date
    mask = date_index != last_date
    X = X[mask]
    y = y[mask]

if use_best_params:
    model_params['random_state'] = RANDOM_SEED  # Add this line
    model_params.update(best_params)
    model = lgb.LGBMClassifier(**model_params)
else:
    # Ensure feature_pre_filter is False in base model params to allow min_data_in_leaf changes
    model_params['feature_pre_filter'] = False
    model = lgb.LGBMClassifier(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=15,
        **model_params
    )

cv = MultipleTimeSeriesCV(
        n_splits=CV_SPLITS*CV_PURGE,
        train_period_length=YEAR * YEARS_TRAIN,
        test_period_length=WEEKS_TEST,
        lookahead=1,
        date_idx='date',
        shuffle=False,  # Verify this is False for reproducibility
        )  # Add if shuffle=True is needed

if backtest_to_date:
    backtest_end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    backtest_date_values = pd.to_datetime(X.index.get_level_values('date').unique())
    if pd.to_datetime(backtest_end_date) not in backtest_date_values:
        backtest_end_date = backtest_date_values.max().strftime('%Y-%m-%d')



#%% Data Preprocessing
import optuna


valid_results = []
valid_df_saved = []
test_df_saved = []
meta_df_saved = []

backtest_dates = pd.date_range(start=backtest_start_date, 
                              end=backtest_end_date, 
                              freq=backtest_frequency)

model_pre = None
train_df_pre = None
valid_df_pre = None

PERIODS_RECOMPUTE_PARAMS = len(backtest_dates) // 4
# PERIODS_RECOMPUTE_PARAMS = 3
if ENABLE_OPTIMIZATION:
    # Create a refined hyperparameter space
    HYPERPARAMETER_SPACE_FINE = create_refined_hyperparameter_space(HYPERPARAMETER_SPACE, 
                                                                    model_params, 20)
for test_date in tqdm(backtest_dates, desc="Backtesting progress", unit="date"):
    train_df, valid_df, X_test, y_test = create_lgbm_dataset(X, y,
                                            lookback_period=YEARS_TRAIN * 52,
                                            n_test_periods=WEEKS_VALIDATION,
                                            end_date=test_date,
                                            sample_weights=sample_weights,
                                            categorical_features=CATEGORICAL_FEATURES)

    if train_df is None or valid_df is None:
        print(f"Skipping {test_date}: Not enough data for train/valid split.")
        continue

    if retrain_model or len(valid_results) == 0:
        # Full retrain
        if valid_df is not None and hasattr(valid_df, 'data') and len(valid_df.data) > 0:
            model.fit(train_df.data, train_df.label,
                     eval_set=[(valid_df.data, valid_df.label)],
                     callbacks=[lgb.early_stopping(50, verbose=False)],
                     categorical_feature=CATEGORICAL_FEATURES,
                     eval_metric =average_precision_eval_sklearn)
            valid_df_saved.append(valid_df)
        else:
            model.fit(train_df.data, train_df.label,
                     categorical_feature=CATEGORICAL_FEATURES)
        model_save = copy.deepcopy(model)
    elif incremental_training:
        # Incremental training on new data only
        model.fit(train_df.data, train_df.label,
                 init_model=model,
                 categorical_feature=CATEGORICAL_FEATURES)
        model_save = copy.deepcopy(model)

    # Make predictions
    valid_pred = model.predict(X_test)
    valid_pred_proba = model.predict_proba(X_test)[:, 1]

    if len(test_df_saved) > 1:

        concatenated_valid_df = lgb.Dataset(
            np.vstack([df.data for df in test_df_saved[:-1]]),
            label=np.concatenate([df.label for df in test_df_saved[:-1]]),
            categorical_feature=CATEGORICAL_FEATURES
        )
        # Extract and concatenate pandas indices from test_df_saved
        all_indices = pd.concat([pd.DataFrame(index=df.pandas_index) for df in test_df_saved[:-1]]).index
        # Store the concatenated pandas index
        concatenated_valid_df.pandas_index = all_indices

    # if model_pre is not None:
        # Calibrate the model
        calibrator = CalibratedClassifierCV(model_pre,
                                            method='isotonic',
                                            cv='prefit',
                                            )  # Add random_state
        
        # Fit calibrator on just the training data
        calibrator.fit(concatenated_valid_df.data, concatenated_valid_df.label)

        # Make calibrated predictions and probabilities
        calibrator_pred = calibrator.predict(X_test)
        calibrator_pred_proba = calibrator.predict_proba(X_test)[:, 1] # Get probability for the positive class


        concatenated_meta = lgb.Dataset(
            np.vstack([df.data for df in test_df_saved[:-1]]),
            label=np.concatenate([np.logical_and(df['true_labels'] == 1,
                                                  df['predictions'] == 1) for df in valid_results[:-1]]),
            categorical_feature=CATEGORICAL_FEATURES
        )
        # Extract and concatenate pandas indices from test_df_saved
        all_indices = pd.concat([pd.DataFrame(index=df.pandas_index) for df in test_df_saved[:-1]]).index
        # Store the concatenated pandas index
        concatenated_meta.pandas_index = all_indices

        meta_labeler = lgb.LGBMClassifier(max_depth=4, n_iterations=100, learning_rate=0.05, is_unbalance = True,
                                          boosting_type='rf', random_state=RANDOM_SEED, metric = 'average_precision_score', 
                                          colsample_bytree=0.7, subsample=0.7)
        meta_labeler.fit(pd.DataFrame(concatenated_meta.data, index=concatenated_meta.pandas_index,
                                      columns=X.columns).assign(
                            preds=np.concatenate([df['predictions_proba'] for df in valid_results[:-1]])).astype(
                                {'weekofyear':'int','sector': 'int'}).astype(
                                {'weekofyear':'category','sector': 'category'}), 
                         pd.Series(concatenated_meta.label, index=concatenated_meta.pandas_index),
                         categorical_feature=CATEGORICAL_FEATURES)
        
        meta_pred = meta_labeler.predict(X_test.assign(
                            preds=valid_pred_proba))
        meta_pred_proba = meta_labeler.predict_proba(X_test.assign(
                            preds=valid_pred_proba))[:, 1]

    else:
        # If no previous model, use the current model's predictions
        calibrator_pred = valid_pred
        calibrator_pred_proba = valid_pred_proba
        meta_pred = valid_pred
        meta_pred_proba = valid_pred_proba

    # Store previous model and data if needed for analysis or comparison

    model_pre = copy.deepcopy(model)
    train_df_pre = copy.deepcopy(train_df)
    valid_df_pre = copy.deepcopy(valid_df)
    # Store results
    if len(np.unique(y_test)) == 1:
        # Handle cases where only one class is present in the validation set
        result = {
            'date': test_date,
            'predictions': valid_pred,
            'predictions_proba': valid_pred_proba,
            'true_labels': y_test.values,
            'calibrator_predictions': calibrator_pred,
            'calibrator_predictions_proba': calibrator_pred_proba, # Added calibrated probabilities
            'validation_auc': np.nan, # Set to None when only one class is present
            'validation_ap': np.nan,  # Set to None when only one class is present
            'meta_predictions': meta_pred,
            'meta_predictions_proba': meta_pred_proba,  # Added meta predictions
        }
    else:
        # Standard case with multiple classes
        result = {
            'date': test_date,
            'predictions': valid_pred,
            'predictions_proba': valid_pred_proba,
            'true_labels': y_test.values,
            'calibrator_predictions': calibrator_pred,
            'calibrator_predictions_proba': calibrator_pred_proba, # Added calibrated probabilities
            'validation_auc': roc_auc_score(y_test, valid_pred_proba), # Use probabilities for AUC
            'validation_ap': average_precision_score(y_test, valid_pred_proba), # Use probabilities for AP
            'meta_predictions': meta_pred,  # Added meta predictions
            'meta_predictions_proba': meta_pred_proba,  # Added meta predictions
        }
    valid_results.append(result)
    # Create Dataset with the data and label, then store the index separately
    dataset = lgb.Dataset(X_test, label=y_test, categorical_feature=CATEGORICAL_FEATURES)
    dataset.pandas_index = X_test.index  # Store the index as a custom attribute
    test_df_saved.append(dataset)

    dataset_meta = lgb.Dataset(X_test.assign(preds=valid_pred_proba), label=np.logical_and(y_test,valid_pred), 
                               categorical_feature=CATEGORICAL_FEATURES)
    dataset_meta.pandas_index = X_test.index
    meta_df_saved.append(dataset_meta)

    # Check if it's time to recompute hyperparameters
    if ENABLE_OPTIMIZATION and len(valid_results) > 0 and len(valid_results) % PERIODS_RECOMPUTE_PARAMS == 0:
        CV_START_DATE = pd.Timestamp(test_date, tz='UTC') - pd.DateOffset(years=4*YEARS_TRAIN)

        date_mask = ((X.index.get_level_values('date') >= CV_START_DATE) & 
             (X.index.get_level_values('date') < pd.Timestamp(test_date, tz='UTC')))

        # Create CV datasets with date filter
        X_cv = X.loc[date_mask]
        y_cv = y.loc[date_mask]
        def objective(trial):
            params = parse_hyperparameter_space(trial, HYPERPARAMETER_SPACE_FINE)
            # Add objective parameter to use custom_logloss
            fold_scores = []

            for train_idx, test_idx in purge_cv_folds(cv.split(X_cv),CV_PURGE):
                X_train, X_test = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
                y_train, y_test = y_cv.iloc[train_idx], y_cv.iloc[test_idx]

                weights_train = sample_weights.loc[X_train.index].abs() if USE_RETURN_AS_WEIGHT else None
                weights_test = sample_weights.loc[X_test.index].abs() if USE_RETURN_AS_WEIGHT else None

                dtrain = lgb.Dataset(X_train, label=y_train,
                                    categorical_feature = CATEGORICAL_FEATURES, weight=weights_train)
                dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain,
                                    categorical_feature = CATEGORICAL_FEATURES, weight=weights_test)

                bst = lgb.train(
                    params,
                    dtrain,
                    valid_sets=[dtest],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                    feval=[average_precision_eval if HYPERPARAMETER_SPACE['boosting_type'][0] == 'average_precision_score' else auc_feval],
                    # init_model = model 
                )
                metric = average_precision_score if HYPERPARAMETER_SPACE['boosting_type'][0] == 'average_precision_score' else roc_auc_score

                # Compute metrics
                preds = bst.predict(X_test, num_iteration=bst.best_iteration)
                aps = metric(y_test, preds, sample_weight = weights_test) if USE_RETURN_AS_WEIGHT else metric(y_test, preds)
                fold_scores.append(aps)


            return np.mean(fold_scores)

        # Add a seeded sampler
        sampler = optuna.samplers.TPESampler(multivariate=True, seed=RANDOM_SEED)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.enqueue_trial(model_params)
        # We need to pass the train and validation datasets to the objective function
        study.optimize(objective, n_trials=4, show_progress_bar=True)
        best_params = study.best_params
        model_params.update(best_params)
        model = lgb.LGBMClassifier(**model_params)
        print(f"Updated model parameters: {model_params}")

# Concatenate test_df_saved datasets while preserving pandas index
if test_df_saved:
    # Extract data, labels, and indices
    all_data = np.vstack([df.data for df in test_df_saved])
    all_labels = np.concatenate([df.label for df in test_df_saved])
    all_indices = pd.concat([pd.DataFrame(index=df.pandas_index) for df in test_df_saved]).index
    
    # Create the concatenated dataset
    concatenated_valid_df = lgb.Dataset(
        all_data,
        label=all_labels,
        categorical_feature=CATEGORICAL_FEATURES
    )
    
    # Store the concatenated pandas index
    concatenated_valid_df.pandas_index = all_indices
else:
    concatenated_valid_df = None

# Concatenate meta_df_saved datasets while preserving pandas index
if meta_df_saved:
    # Extract data, labels, and indices
    all_meta_data = np.vstack([df.data for df in meta_df_saved])
    all_meta_labels = np.concatenate([df.label for df in meta_df_saved])
    all_meta_indices = pd.concat([pd.DataFrame(index=df.pandas_index) for df in meta_df_saved]).index
    
    # Create the concatenated meta dataset
    concatenated_meta_df = lgb.Dataset(
        all_meta_data,
        label=all_meta_labels,
        categorical_feature=CATEGORICAL_FEATURES
    )
    
    # Store the concatenated pandas index
    concatenated_meta_df.pandas_index = all_meta_indices
else:
    concatenated_meta_df = None



# %% Post-Backtesting Analysis

# Load returns data needed for subsequent analyses
returns_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '')),
                                      "../output/target.csv"), index_col=0, parse_dates=True)
returns_df.index = pd.to_datetime(returns_df.index) # Ensure index is datetime

# Get unique sectors from the data index
sectors = sorted(X.index.get_level_values('ticker').unique())

# Analyze prediction accuracy for both original and calibrated predictions
prediction_accuracy_df = analyze_predictions_accuracy(valid_results, sectors,  prediction_type="normal")
prediction_accuracy_df_calibrated = analyze_predictions_accuracy(valid_results, sectors, prediction_type="calibrator")
prediction_accuracy_df_meta = analyze_predictions_accuracy(valid_results, sectors, prediction_type="meta")

# Localize the index timezone to UTC for consistency
prediction_accuracy_df.index = prediction_accuracy_df.index.tz_localize('UTC')
prediction_accuracy_df_calibrated.index = prediction_accuracy_df_calibrated.index.tz_localize('UTC')
prediction_accuracy_df_meta.index = prediction_accuracy_df_meta.index.tz_localize('UTC')

# Extract true labels for each sector
true_labels_df = extract_true_labels(valid_results, sectors)

# Extract plain predictions for each sector
prediction_df = extract_predictions(valid_results, sectors, prediction_type="normal")
calibrator_prediction_df = extract_predictions(valid_results, sectors, prediction_type="calibrator")
meta_prediction_df = extract_predictions(valid_results, sectors, prediction_type="meta")

# Extract prediction probabilities for each sector
prediction_proba_df = extract_predictions_proba(valid_results, sectors, prediction_type="normal")
calibrator_prediction_proba_df = extract_predictions_proba(valid_results, sectors, prediction_type="calibrator")
meta_prediction_proba_df = extract_predictions_proba(valid_results, sectors, prediction_type="meta")

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
correlation_data_path = save_dataframe_to_csv(correlation_data, config_path, "correlation_data.csv")

# %% Saving results
# Call the function to save valid DataFrames
valid_df_path = save_valid_dfs(test_df_saved, X, config_path)

# Save top sector analysis
top_sector_path = save_dataframe_to_csv(top_sector_analysis, config_path, "top_sector_analysis.csv")

# Save gains analysis
gains_path = save_dataframe_to_csv(
    {"total_gains": total_gains, "potential_gains": potential_gains},
    config_path, 
    "gains_analysis.csv"
)

# Save model
model_path = save_model(model_save, config_path)

# Save feature importances
importance_path = save_dataframe_to_csv(
    pd.DataFrame({
        'feature': X.columns,
        'importance': model_save.feature_importances_
    }).sort_values('importance', ascending=False),
    config_path,
    "feature_importances.csv",
    index=False
)

# Save model parameters
params_path = save_dataframe_to_csv(
    pd.DataFrame.from_dict(model_save.get_params(), orient='index', columns=['value']),
    config_path,
    "model_parameters.csv"
)

# Save prediction accuracy
accuracy_paths = save_dataframe_to_csv(
    {
        "original": prediction_accuracy_df,
        "calibrated": prediction_accuracy_df_calibrated,
        "meta": prediction_accuracy_df_meta
    },
    config_path,
    {
        "original": "prediction_accuracy.csv",
        "calibrated": "prediction_accuracy_calibrated.csv",
        "meta": "prediction_accuracy_meta.csv"
    }
)

# Save categorical info
cat_info_path = save_categorical_info(CATEGORICAL_FEATURES if CATEGORICAL_FEATURES != "auto" else [], valid_df_path)

# Save plain predictions
prediction_paths = save_dataframe_to_csv(
    {
        "model": prediction_df,
        "calibrator": calibrator_prediction_df,
        "meta": meta_prediction_df,
        "true_labels": true_labels_df
    },
    config_path,
    {
        "model": "model_predictions.csv",
        "calibrator": "calibrator_predictions.csv",
        "meta": "meta_predictions.csv",
        "true_labels": "true_labels.csv"
    }
)

# Save prediction probabilities
proba_paths = save_dataframe_to_csv(
    {
        "original": prediction_proba_df,
        "calibrated": calibrator_prediction_proba_df,
        "meta": meta_prediction_proba_df
    },
    config_path,
    {
        "original": "prediction_proba.csv",
        "calibrated": "calibrator_prediction_proba.csv",
        "meta": "meta_prediction_proba.csv"
    }
)