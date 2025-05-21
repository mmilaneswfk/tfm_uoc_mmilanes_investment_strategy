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
    create_time_encoding_rbf, simple_labeling,
    create_diff, last_target_outcomes, create_cyclical_features,
    std_labeling, labeling_selector, create_cross_sectional_feature)
from src.functions.optimization_functions import purge_cv_folds, parse_hyperparameter_space, custom_logloss, average_precision_eval, auc_feval
from src.functions.feature_selection_functions import BorutaShap
from sklearn.metrics import roc_auc_score, average_precision_score
import datetime
import optuna
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
import random

RANDOM_SEED = 42  # Set random seed for reproducibility

# Set random seeds for all libraries
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

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
    COVID_FILTER = config['COVID_FILTER']

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
    OPTIMIZATION_TRIALS = config['OPTIMIZATION_TRIALS']
    COLUMNS_TO_KEEP = config['COLUMNS_TO_KEEP']
    COLUMNS_TO_DROP = config['COLUMNS_TO_DROP']
    USE_SELECTION_IF_AVAILABLE = config['USE_SELECTION_IF_AVAILABLE']
    KEEP_TIME_FEATURES = config['KEEP_TIME_FEATURES']
    USE_RETURN_AS_WEIGHT = config['USE_RETURN_AS_WEIGHT']

config_hyper_path = os.path.join(os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '')),
                           '../configs/hyperparameters.yml')

with open(config_hyper_path, 'r') as file:
    config_hyper = yaml.safe_load(file)
    HYPERPARAMETER_SPACE = config_hyper['HYPERPARAMETER_SPACE']

idx = pd.IndexSlice

# Define key time periods
CV_START_DATE = pd.Timestamp(VALIDATION_DATE, tz='UTC') - pd.DateOffset(years=3*YEARS_TRAIN)
VALIDATION_TIMESTAMP = pd.Timestamp(VALIDATION_DATE, tz='UTC')
COVID_START_DATE = pd.Timestamp('2020-02-20', tz='UTC')
COVID_END_DATE = pd.Timestamp('2020-06-30', tz='UTC')

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
# Apply standardization to each column (z-score normalization)
for col in weights_unstacked.columns:
    weights_date_mask = ((weights_unstacked.index >= CV_START_DATE) & 
                         (weights_unstacked.index < VALIDATION_TIMESTAMP))
    mean_val = weights_unstacked.loc[weights_date_mask, col].mean()
    std_val = weights_unstacked.loc[weights_date_mask, col].std()
    if std_val > 0:  # Avoid division by zero
        weights_unstacked[col] = (weights_unstacked[col] - mean_val) / std_val

weights_unstacked = weights_unstacked.abs().sort_index()

weights_unstacked.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '')),
                                        '../output/weights.csv'), index=True)
weights_stacked = weights_unstacked.stack()
weights_stacked.index = weights_stacked.index.swaplevel()
weights_stacked = weights_stacked.sort_index()

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
# data = create_time_encoding_rbf(data, TIME_FEATURES, 4)
data = create_diff(data, DIFF)
data = last_target_outcomes(data, TARGET_NAME, TARGET_THRESHOLD)
data = create_cyclical_features(data, ['month', 'weekofyear'])
data = create_cross_sectional_feature(data, TARGET_NAME, 'std', 1, 'fe')
data = create_cross_sectional_feature(data, TARGET_NAME, 'mean', 1, 'fe')

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

#%% Data splitting and preparation
# Split data into features (X) and target (y)
y = data[TARGET_NAME]
X = data.drop(TARGET_NAME, axis=1)

# Apply COVID filter if enabled
if COVID_FILTER:
    print(f"Filtering out COVID period: {COVID_START_DATE.date()} to {COVID_END_DATE.date()}")
    covid_mask = ~((X.index.get_level_values('date') >= COVID_START_DATE) & 
                  (X.index.get_level_values('date') <= COVID_END_DATE))
    
    # Apply filter to all relevant datasets
    X = X.loc[covid_mask]
    y = y.loc[covid_mask]
    # Create COVID mask for weights using its own index
    weights_covid_mask = ~((weights_stacked.index.get_level_values('date') >= COVID_START_DATE) & 
                          (weights_stacked.index.get_level_values('date') <= COVID_END_DATE))
    weights_stacked = weights_stacked.loc[weights_covid_mask]

# Create date filter for CV data
date_mask = ((X.index.get_level_values('date') >= CV_START_DATE) & 
             (X.index.get_level_values('date') < VALIDATION_TIMESTAMP))

# Apply binary labeling to target variable
#y = simple_labeling(y, TARGET_THRESHOLD)
y = labeling_selector(y, 'simple')

# Create CV datasets with date filter
X_cv = X.loc[date_mask]
y_cv = y.loc[date_mask]

print(f"CV dataset: {X_cv.shape[0]} samples from {CV_START_DATE.date()} to {VALIDATION_TIMESTAMP.date()}")

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
                                    max_depth=8,
                                    num_leaves=25,
                                    subsample=0.7,
                                    colsample_bytree=0.8,
                                    force_col_wise = True,
                                    boosting_type=HYPERPARAMETER_SPACE['boosting_type'][0],
                                    extra_trees=HYPERPARAMETER_SPACE['extra_trees'][0],
                                    objective=HYPERPARAMETER_SPACE['objective'][0],
                                    metric=HYPERPARAMETER_SPACE['metric'][0],
                                    learning_rate=0.05,
                                    is_unbalance=HYPERPARAMETER_SPACE['is_unbalance'][0],
                                    random_state=RANDOM_SEED,  # Add this
                                    verbose = -1,),
                        importance_measure='shap',
                        classification=True,
                        percentile=80)

    # Fit the selector
    weights = weights_stacked.loc[X_cv.dropna().index].abs() if USE_RETURN_AS_WEIGHT else None
    feature_selector.fit(X=X_cv.dropna(), y=y_cv.loc[X_cv.dropna().index], 
                         sample_weight=weights,
                         categorical_feature = CATEGORICAL_FEATURES,
                        n_trials=50, sample=True,
                        train_or_test = 'test', normalize=False, random_state=RANDOM_SEED,
                verbose=True)
    # Get selected features

    selected_features = feature_selector.Subset(tentative=True).columns
    # feature_selector.plot(which_features='all')

    # Print results
    print(f"Number of selected features: {len(selected_features)}")
    print("Selected features:", selected_features)
# Start with the base selected features from BorutaShap
selected_features = list(selected_features)

# Add required features that might be missing
missing_required = [col for col in COLUMNS_TO_KEEP if col not in selected_features]
if missing_required:
    print(f"Adding {len(missing_required)} required columns")
    selected_features.extend(missing_required)

# Add time-related features if specified
if KEEP_TIME_FEATURES:
    # Add cyclical time features (sin/cos)
    sin_cos_features = [col for col in X.columns if (col.endswith('_sin') or col.endswith('_cos')) 
                        and col not in selected_features]
    # Add RBF time encodings
    time_rbf_features = [col for col in X.columns if col.startswith('time_rbf_') 
                         and col not in selected_features]
    
    time_features = sin_cos_features + time_rbf_features
    if time_features:
        print(f"Adding {len(time_features)} time-related features")
        selected_features.extend(time_features)

# Remove specified columns to drop
if COLUMNS_TO_DROP:
    before_count = len(selected_features)
    selected_features = [col for col in selected_features if col not in COLUMNS_TO_DROP]
    removed = before_count - len(selected_features)
    if removed > 0:
        print(f"Removed {removed} columns specified in COLUMNS_TO_DROP")

# Check for correlated features
print("\n==== Checking for Correlated Features ====")
X_cv_filtered = X_cv[X_cv.index.get_level_values('date') >= pd.Timestamp('2009-01-01', tz='UTC')]
print(f"Computing correlation matrix for {len(selected_features)} features...")

# Calculate correlation matrix
corr_matrix = X_cv_filtered[selected_features].corr(method='spearman').abs()

# Find and remove highly correlated features
features_to_remove = set()
for i, feature_i in enumerate(selected_features):
    for feature_j in selected_features[i+1:]:
        corr_value = corr_matrix.loc[feature_i, feature_j]
        if np.isnan(corr_value):
            continue
        if corr_value > 0.8:
            print(f"  High correlation ({corr_value:.2f}): keeping '{feature_i}', removing '{feature_j}'")
            features_to_remove.add(feature_j)

if KEEP_TIME_FEATURES:
    # Remove time-related features from the list of features to remove
    features_to_remove = [f for f in features_to_remove if f not in time_features]

if features_to_remove:
    print(f"Removing {len(features_to_remove)} highly correlated features")
    selected_features = [f for f in selected_features if f not in features_to_remove]
else:
    print("No highly correlated features found")

# Final verification for required columns
missing_required = [col for col in COLUMNS_TO_KEEP if col not in selected_features]
if missing_required:
    print(f"Re-adding {len(missing_required)} required columns that were removed during correlation check")
    selected_features.extend(missing_required)

print(f"\nFinal feature count: {len(selected_features)} features")

# Save selected features to model_data.h5
with pd.HDFStore(DATA_STORE) as store:
    store.put('selected_features', pd.Series(selected_features), format='table', data_columns=True)

X = X[selected_features]
X_cv = X_cv[selected_features]

#%% Check for missing values
# Analyze missing values by year and column
print("==== Missing Value Analysis ====")

# 1. Count dates with missing values by year
# Create a Series indicating if a date has any null values across any stock
has_null_by_date = X_cv.groupby(level='date').apply(lambda x: x.isna().any().any())

# Group by year and calculate statistics
print("\n1. Dates with missing values by year:")
# Check for duplicate dates
unique_dates = has_null_by_date.index.nunique()
if len(has_null_by_date) != unique_dates:
    print(f"  Warning: Found {len(has_null_by_date) - unique_dates} duplicate dates")
    # Remove duplicates if any
    has_null_by_date = has_null_by_date[~has_null_by_date.index.duplicated()]

# Create lists of dates with and without nulls
dates_with_nulls = has_null_by_date[has_null_by_date].index
dates_without_nulls = has_null_by_date[~has_null_by_date].index

print(f"  Dates with nulls: {len(dates_with_nulls)} dates")
print(f"  Dates without nulls: {len(dates_without_nulls)} dates")
print(f"  Percentage of dates with nulls: {(len(dates_with_nulls) / len(has_null_by_date)) * 100:.1f}%")

# Get years with at least one date having nulls
years_with_nulls = sorted(set(d.year for d in dates_with_nulls))
# Get years where no date has nulls
all_years = sorted(set(d.year for d in has_null_by_date.index))
years_without_nulls = sorted(set(all_years) - set(years_with_nulls))

print(f"  Years with nulls: {', '.join(map(str, years_with_nulls))}")
print(f"  Years without nulls: {', '.join(map(str, years_without_nulls))}")

# 2. Calculate columns with most nulls
null_counts = X_cv.isna().sum()
null_percent = (null_counts / len(X_cv)) * 100
top_null_columns = null_percent.sort_values(ascending=False).head(5)

print("\n2. Top 5 columns with highest percentage of nulls:")
for col, pct in top_null_columns.items():
    count = null_counts[col]
    print(f"  {col}: {count} nulls ({pct:.2f}%)")

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
    # Add seed parameter
    params['seed'] = RANDOM_SEED
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
            feval=[average_precision_eval if HYPERPARAMETER_SPACE['boosting_type'][0] == 'average_precision_score' else auc_feval]  # Add fobj parameter for custom objective
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
study.optimize(objective, n_trials=OPTIMIZATION_TRIALS, show_progress_bar=True)


best_params = study.best_params
print(f"Best params: {best_params}")

# Add seed to best_params
best_params['seed'] = RANDOM_SEED

# Save preprocessed data to model_data.h5
with pd.HDFStore(DATA_STORE) as store:
    store.put('best_params', pd.Series(best_params), format='table', data_columns=True)

#%%
# Create a sample using only the last years of data
max_date = X_cv.index.get_level_values('date').max()
train_years_ago = max_date - pd.DateOffset(years=YEARS_TRAIN)
date_sample_filter = X_cv.index.get_level_values('date') >= train_years_ago

# Create sample datasets
X_sample = X_cv[date_sample_filter]
y_sample = y_cv[date_sample_filter]

# Train final model with best parameters on sample data
weights_tdata = weights_stacked.loc[X_sample.dropna().index].abs() if USE_RETURN_AS_WEIGHT else None
dtrain = lgb.Dataset(X_sample.dropna(), label=y_sample.loc[X_sample.dropna().index], categorical_feature=CATEGORICAL_FEATURES,
                     weight=weights_tdata)
final_model = lgb.train(best_params, dtrain)

# Get feature importances
importances = pd.DataFrame({
    'feature': X_sample.columns,
    'importance': final_model.feature_importance(importance_type='gain')
})
importances = importances.sort_values('importance', ascending=False)

# Print top 5 features
# Convert importances to percentage of total importance
importances['importance_pct'] = importances['importance'] / importances['importance'].sum() * 100

print("\nTop 5 most important features (% of total importance):")
print(importances[['feature', 'importance_pct']].head())


# %%
