#%% Imports and config
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import yaml
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.utils import MultipleTimeSeriesCV
from src.functions.feature_engineering_functions import (create_lags, create_rolling_features, 
    create_momentum_features, create_time_features, 
    create_time_encoding_rbf, simple_labelling,
    create_diff, last_target_outcomes)
from src.functions.optimization_functions import purge_cv_folds, parse_hyperparameter_space, custom_logloss, average_precision_eval
from src.functions.feature_selection_functions import BorutaShap
from sklearn.metrics import roc_auc_score, average_precision_score
import datetime
import optuna
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

RANDOM_SEED = 42 # Set random seed for reproducibility

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '')),
                           '../configs/modeling_config.yml')

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    # Data paths and storage
    DATA_STORE = config['DATA_STORE']

    # Time period configurations
    YEAR = config['YEAR']
    YEARS_TRAIN = config['YEARS_TRAIN']
    WEEKS_TEST = config['WEEKS_TEST']
    VALIDATION_DATE = config['VALIDATION_DATE']

    # Cross-validation parameters
    CV_SPLITS = config['CV_SPLITS']
    CV_PURGE = config['CV_PURGE']

    # Target and threshold
    TARGET_NAME = config['TARGET_NAME']
    TARGET_THRESHOLD = config['TARGET_THRESHOLD']

    # Feature engineering parameters
    LAGS = config['LAGS']
    ROLLING = config['ROLLING']
    MOMENTUM = config['MOMENTUM']
    DIFF = config['DIFF']
    TIME_FEATURES = config['TIME_FEATURES']
    CATEGORICAL_FEATURES = config['CATEGORICAL_FEATURES']
    FAMA_FRENCH_FACTORS = config['FAMA_FRENCH_FACTORS']

    # Model and feature selection settings
    HYPERPARAMETER_SPACE = config['HYPERPARAMETER_SPACE']
    OPTIMIZATION_TRIALS = config['OPTIMIZATION_TRIALS']
    COLUMNS_TO_KEEP = config['COLUMNS_TO_KEEP']
    COLUMNS_TO_DROP = config['COLUMNS_TO_DROP']
    USE_SELECTION_IF_AVAILABLE = config['USE_SELECTION_IF_AVAILABLE']
    KEEP_TIME_FEATURES = config['KEEP_TIME_FEATURES']
    USE_RETURN_AS_WEIGHT = config['USE_RETURN_AS_WEIGHT']

idx = pd.IndexSlice

DATA_STORE = os.path.join(os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '')), 
                          DATA_STORE)

if not CATEGORICAL_FEATURES:
    CATEGORICAL_FEATURES = "auto"
if not FAMA_FRENCH_FACTORS:
    FAMA_FRENCH_FACTORS = []

with pd.HDFStore(DATA_STORE) as store:
    data = (store['engineered_features']
            .sort_index())  # train & validation period

#%% Data Preprocessing

# Unstack returns and save to CSV
target_unstacked = data[TARGET_NAME].unstack(level=0)
target_unstacked.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '')),
                                        '../output/target.csv'), index=True)

weights_unstacked = data[TARGET_NAME].fillna(0).unstack(level=0)
weights_unstacked.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '')),
                                        '../output/weights.csv'), index=True)
weights_stacked = weights_unstacked.stack()
weights_stacked.index = weights_stacked.index.swaplevel()

# Remove any columns that start with 'target_'
data = data.loc[:, ~(data.columns.str.startswith('target_') & (data.columns != TARGET_NAME))]
data = data.sort_index(level=[0, 1])

# Feature Engineering
data = create_lags(data, TARGET_NAME, LAGS)
data = create_rolling_features(data, TARGET_NAME, ROLLING)

for col in FAMA_FRENCH_FACTORS:
    data = create_rolling_features(data, col, ROLLING, prefix=f'ff_{col}_')
    data = create_lags(data, col, LAGS, prefix=f'ff_{col}_')

data = create_momentum_features(data, MOMENTUM)
data = create_time_features(data, TIME_FEATURES.keys())
data = create_time_encoding_rbf(data, TIME_FEATURES, 6)
data = create_diff(data, DIFF)
data = last_target_outcomes(data, TARGET_NAME, TARGET_THRESHOLD)

# Keep only columns with 'return' that contain 't-' followed by a number, or if they match TARGET_NAME
data = data.loc[:, ~(data.columns.str.startswith('return') & 
                    #  (~data.columns.str.contains('t-\d+')) & 
                     (data.columns != TARGET_NAME))]

data = data.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN

# Process categorical features - convert to integer codes and set as category type
if CATEGORICAL_FEATURES != "auto":
    # Check for missing categorical features upfront
    missing_cats = [cat for cat in CATEGORICAL_FEATURES if cat not in data.columns]
    if missing_cats:
        raise ValueError(f"Categorical features not found in the data: {missing_cats}")
    
    # Process all categorical features at once
    for cat in CATEGORICAL_FEATURES:
        # Factorize converts to integer codes (0,1,2...) and set category dtype
        data[cat] = pd.factorize(data[cat], sort=True)[0]
        data[cat] = data[cat].astype('category')


# Save preprocessed data to model_data.h5
with pd.HDFStore(DATA_STORE) as store:
    store.put('modeling_data', data, format='table', data_columns=True)

# Drop specified columns
if COLUMNS_TO_DROP:
    columns_to_drop = [col for col in COLUMNS_TO_DROP if col in data.columns]
    if columns_to_drop:
        print(f"Dropping {len(columns_to_drop)} columns: {columns_to_drop}")
        data = data.drop(columns=columns_to_drop)
    else:
        print("No columns to drop found in data")

#%% Spliting the data
# Split data into features (X) and target (y)
# Create date filter condition
date_filter = (
    (data.index.get_level_values('date') < pd.Timestamp(VALIDATION_DATE, tz='UTC')) & 
    (data.index.get_level_values('date') >= pd.Timestamp(VALIDATION_DATE, tz='UTC') - pd.DateOffset(years=3*YEARS_TRAIN))
)

# Split data into features (X) and target (y)
y = data[TARGET_NAME]
X = data.drop(TARGET_NAME, axis=1)

# Check if COVID_FILTER exists in config, otherwise set default
COVID_FILTER = config.get('COVID_FILTER', False)  # Default to False if not in config

# Define COVID period dates (most impactful market period)
COVID_START_DATE = pd.Timestamp('2020-02-20', tz='UTC')  # Beginning of market crash
COVID_END_DATE = pd.Timestamp('2020-06-30', tz='UTC')    # Most severe market impact period

# Filter out COVID period if enabled
if COVID_FILTER:
    print(f"Filtering out COVID period from {COVID_START_DATE.date()} to {COVID_END_DATE.date()}")
    covid_filter = ~((data.index.get_level_values('date') >= COVID_START_DATE) & 
                     (data.index.get_level_values('date') <= COVID_END_DATE))
    data = data.loc[covid_filter]
    X = X.loc[covid_filter]
    y = y.loc[covid_filter]
    
    # Also update weights if they're being used
    if USE_RETURN_AS_WEIGHT:
        weights_stacked = weights_stacked.loc[covid_filter]

# Labeling
y = simple_labelling(y, TARGET_THRESHOLD)

# Apply date filter to create CV datasets
X_cv = X[date_filter]
y_cv = y[date_filter]

#%% Feature selection
# Initialize BorutaShap for feature selection

selected_features = None

if USE_SELECTION_IF_AVAILABLE:
    try:
        with pd.HDFStore(DATA_STORE) as store:
            if 'selected_features' in store:
                selected_features = store['selected_features'].values
                print(f"Loaded {len(selected_features)} pre-selected features from storage")
    except Exception as e:
        print(f"Error loading saved features: {e}")

if selected_features is None:
    print("Performing feature selection...")
    feature_selector = BorutaShap(model=lgb.LGBMClassifier(
                                    num_iterations=100,
                                    n_estimators=200,
                                    max_depth=150,
                                    num_leaves=50,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    min_child_samples=60,
                                    force_col_wise = True,
                                    boosting_type='rf',
                                    extra_trees=True,
                                    objective='binary',
                                    metric='average_precision_score',
                                    learning_rate=0.5,
                                    is_unbalance=True,
                                    verbose = -1,),
                        importance_measure='shap',
                        classification=True,
                        percentile=85)

    # Fit the selector
    weights = weights_stacked.loc[X_cv.dropna().index].abs() if USE_RETURN_AS_WEIGHT else None
    feature_selector.fit(X=X_cv.dropna(), y=y_cv.loc[X_cv.dropna().index], 
                         sample_weight=weights,
                         categorical_feature = CATEGORICAL_FEATURES,
                        n_trials=50, sample=True,
                        train_or_test = 'test', normalize=True, random_state=RANDOM_SEED,
                verbose=True)
    # Get selected features

    selected_features = feature_selector.Subset(tentative=True).columns
    # feature_selector.plot(which_features='all')

    # Print results
    print(f"Number of selected features: {len(selected_features)}")
    print("Selected features:", selected_features)
selected_features = list(selected_features) + [x for x in COLUMNS_TO_KEEP if x not in selected_features]

if "vix_diff" in selected_features and "vix_chg" in selected_features:
    selected_features.remove("vix_chg")

if KEEP_TIME_FEATURES:
    time_rbf_features = [col for col in X.columns if col.startswith('time_rbf_') and col not in selected_features]
    selected_features.extend(time_rbf_features)
print(selected_features)

# Drop specified columns
if COLUMNS_TO_DROP:
    selected_features = [col for col in selected_features if col not in COLUMNS_TO_DROP]

# Save selected features to model_data.h5
with pd.HDFStore(DATA_STORE) as store:
    store.put('selected_features', pd.Series(selected_features), format='table', data_columns=True)

X = X[selected_features]
X_cv = X_cv[selected_features]

#%% Optimization step
# Create cv split

cv = MultipleTimeSeriesCV(
        n_splits=CV_SPLITS*CV_PURGE,
        train_period_length=YEAR * YEARS_TRAIN,
        test_period_length=WEEKS_TEST,
        lookahead=1,
        date_idx='date',
        shuffle=False)


def objective(trial):
    params = parse_hyperparameter_space(trial, HYPERPARAMETER_SPACE)
    # Add objective parameter to use custom_logloss
    fold_scores = []

    for train_idx, test_idx in purge_cv_folds(cv.split(X_cv),CV_PURGE):
        X_train, X_test = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
        y_train, y_test = y_cv.iloc[train_idx], y_cv.iloc[test_idx]

        weights_train = weights_stacked.loc[X_train.index].abs() if USE_RETURN_AS_WEIGHT else None
        weights_test = weights_stacked.loc[X_test.index].abs() if USE_RETURN_AS_WEIGHT else None

        dtrain = lgb.Dataset(X_train, label=y_train,
                             categorical_feature = CATEGORICAL_FEATURES, weight=weights_train)
        dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain,
                            categorical_feature = CATEGORICAL_FEATURES, weight=weights_test)

        bst = lgb.train(
            params,
            dtrain,
            valid_sets=[dtest],
            callbacks=[lgb.early_stopping(50, verbose=False)],
            feval=average_precision_eval  # Add fobj parameter for custom objective
        )

        # Compute metrics
        preds = bst.predict(X_test, num_iteration=bst.best_iteration)
        aps = average_precision_score(y_test, preds)
        fold_scores.append(aps)

    return np.mean(fold_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=OPTIMIZATION_TRIALS, show_progress_bar=True)


best_params = study.best_params
print(f"Best params: {best_params}")


# Save preprocessed data to model_data.h5
with pd.HDFStore(DATA_STORE) as store:
    store.put('best_params', pd.Series(best_params), format='table', data_columns=True)

#%%
# Train final model with best parameters on all CV data
weights_tdata = weights_stacked.loc[X_cv.dropna().index].abs() if USE_RETURN_AS_WEIGHT else None
dtrain = lgb.Dataset(X_cv.dropna(), label=y_cv.loc[X_cv.dropna().index], categorical_feature = CATEGORICAL_FEATURES,
                     weight=weights_tdata)
final_model = lgb.train(best_params, dtrain)

# Get feature importances
importances = pd.DataFrame({
    'feature': X_cv.columns,
    'importance': final_model.feature_importance(importance_type='gain')
})
importances = importances.sort_values('importance', ascending=False)

# Print top 5 features
# Convert importances to percentage of total importance
importances['importance_pct'] = importances['importance'] / importances['importance'].sum() * 100

print("\nTop 5 most important features (% of total importance):")
print(importances[['feature', 'importance_pct']].head())

