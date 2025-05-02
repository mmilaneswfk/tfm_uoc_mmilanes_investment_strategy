import lightgbm as lgb
import pandas as pd
from typing import List, Tuple
import os
import pickle
import numpy as np

def create_lgbm_dataset(X: pd.DataFrame, y: pd.Series, lookback_period: int, 
                        n_test_periods: int, end_date: str = None,
                        sample_weights: pd.DataFrame = None,
                        categorical_features: List[str] = None
                        ) -> Tuple[lgb.Dataset, lgb.Dataset]:
    """
    Creates LightGBM datasets for training and validation based on date splitting.
    
    Args:
        X (pd.DataFrame): Input features DataFrame with MultiIndex (first level security, second level datetime)
        y (pd.Series): Target variable Series with matching index to X
        lookback_period (int): Number of periods to look back for training
        n_test_periods (int): Number of periods to use for validation
        end_date (str, optional): Filter data up to this date (format: 'YYYY-MM-DD')
    
    Returns:
        tuple: (train_dataset, valid_dataset) LightGBM dataset objects
    """
    # Filter by end_date if provided, otherwise use the latest date in the dataset
    if end_date:
        end_date = pd.to_datetime(end_date)
    else:
        end_date = X.index.get_level_values(1).max()
    
    # Calculate start date
    start_date = end_date - pd.Timedelta(weeks=(lookback_period + n_test_periods))

    # Localize dates in UTC
    if not end_date.tz:
        end_date = end_date.tz_localize('UTC')
    start_date = start_date.tz_localize('UTC')
    
    # Filter DataFrame and Series to include only the desired date range
    mask = (X.index.get_level_values(1) >= start_date) & (X.index.get_level_values(1) <= end_date)
    X = X[mask]
    y = y[mask]
    
    # Get unique dates from second level of index
    dates = X.index.get_level_values(1).unique()
    
    # Calculate split indices
    split_idx = len(dates) - n_test_periods
    start_idx = max(0, split_idx - lookback_period)
    
    # Split data into train and validation sets
    train_dates = dates[start_idx:split_idx]
    valid_dates = dates[split_idx:]
    
    # Filter DataFrame and Series by dates
    X_train = X.loc[pd.IndexSlice[:, train_dates],:]
    y_train = y.loc[pd.IndexSlice[:, train_dates]]
    X_valid = X.loc[pd.IndexSlice[:, valid_dates],:]
    y_valid = y.loc[pd.IndexSlice[:, valid_dates]]

    if sample_weights is not None:
        sample_weights_train = sample_weights.loc[pd.IndexSlice[:, train_dates]]
        sample_weights_valid = sample_weights.loc[pd.IndexSlice[:, valid_dates]]
    else:
        sample_weights_train = np.ones(len(X_train))
        sample_weights_valid = np.ones(len(X_valid))

    # Create LightGBM datasets
    train_dataset = lgb.Dataset(X_train, label=y_train, 
                                weight=sample_weights_train,
                                categorical_feature=categorical_features)
    valid_dataset = lgb.Dataset(X_valid, label=y_valid, reference=train_dataset,
                                weight=sample_weights_valid,
                                categorical_feature=categorical_features)
    # valid_dataset = lgb.Dataset(X_valid, label=y_valid, reference=train_dataset)
    
    return train_dataset, valid_dataset

def flush_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def save_valid_dfs(valid_df_saved, X, config_path):
    valid_df_path = os.path.join(os.path.dirname(config_path), '../output/valid_df')
    os.makedirs(os.path.dirname(valid_df_path), exist_ok=True)
    
    # Create DataFrames with the concatenated data
    data_df = pd.DataFrame(np.vstack([df.data for df in valid_df_saved]), columns=X.columns)
    label_df = pd.DataFrame(np.concatenate([df.label for df in valid_df_saved]), columns=['label'])
    
    data_csv = valid_df_path + '_data.csv'
    label_csv = valid_df_path + '_label.csv'
    dtype_csv = valid_df_path + '_dtypes.csv'
    
    # Flush files before saving
    flush_file(data_csv)
    flush_file(label_csv)
    flush_file(dtype_csv)
    
    data_df.to_csv(data_csv, index=False)
    label_df.to_csv(label_csv, index=False)
    
    # Save metadata for feature dtypes
    dtype_info = {col: str(X[col].dtype) for col in X.columns}
    pd.DataFrame.from_dict(dtype_info, orient='index', columns=['dtype']).to_csv(dtype_csv)
    
    return valid_df_path


def save_model(model, config_path):
    model_path = os.path.join(os.path.dirname(config_path), '../models/lgbm_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Flush file before saving
    flush_file(model_path)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model_path


def save_feature_importances(model, X, config_path):
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = os.path.join(os.path.dirname(config_path), '../output/feature_importances.csv')
    os.makedirs(os.path.dirname(importance_path), exist_ok=True)
    # Flush file before saving
    flush_file(importance_path)
    importance_df.to_csv(importance_path, index=False)
    return importance_path


def save_model_parameters(model, config_path):
    model_params_df = pd.DataFrame.from_dict(model.get_params(), orient='index', columns=['value'])
    params_path = os.path.join(os.path.dirname(config_path), '../output/model_parameters.csv')
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    # Flush file before saving
    flush_file(params_path)
    model_params_df.to_csv(params_path)
    return params_path


def save_prediction_accuracy(accuracy_df, calibrated_accuracy_df, config_path):
    accuracy_path = os.path.join(os.path.dirname(config_path), '../output/prediction_accuracy.csv')
    calibrated_accuracy_path = os.path.join(os.path.dirname(config_path), '../output/prediction_accuracy_calibrated.csv')
    os.makedirs(os.path.dirname(accuracy_path), exist_ok=True)
    # Flush files before saving
    flush_file(accuracy_path)
    flush_file(calibrated_accuracy_path)
    accuracy_df.to_csv(accuracy_path)
    calibrated_accuracy_df.to_csv(calibrated_accuracy_path)
    return accuracy_path, calibrated_accuracy_path


def save_categorical_info(CATEGORICAL_FEATURES, valid_df_path):
    cat_info_path = valid_df_path + '_categorical.csv'
    # Flush file before saving
    flush_file(cat_info_path)
    # Convert list to DataFrame and save as CSV
    pd.DataFrame({'feature_name': CATEGORICAL_FEATURES}).to_csv(cat_info_path, index=False)
    return cat_info_path

def save_top_sector_analysis(top_sector_analysis, config_path):
    """Save top sector analysis results to CSV."""
    output_dir = os.path.join(os.path.dirname(config_path), '../output')
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, 'top_sector_analysis.csv')
    # Flush file before saving
    flush_file(file_path)
    top_sector_analysis.to_csv(file_path)
    return file_path

def save_gains_analysis(total_gains, potential_gains, config_path):
    """Save total and potential gains to CSV."""
    output_dir = os.path.join(os.path.dirname(config_path), '../output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine gains into a DataFrame
    gains_df = pd.DataFrame({
        'total_gains': total_gains,
        'potential_gains': potential_gains
    })
    
    file_path = os.path.join(output_dir, 'gains_analysis.csv')
    # Flush file before saving
    flush_file(file_path)
    gains_df.to_csv(file_path)
    return file_path

def save_correlation_data(correlation_data, config_path):
    base_dir = os.path.dirname(os.path.abspath(config_path))
    output_dir = os.path.join(base_dir, "../output")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "correlation_data.csv")
    flush_file(file_path)
    correlation_data.to_csv(file_path)
    return file_path

def extract_predictions_proba(valid_results, sectors, calibrator=False):
    """
    Build a DataFrame of predicted probabilities for each sector/date.

    valid_results: list of dicts returned in the backtest loop
    sectors: list of sector names in the same order as model inputs
    calibrator: bool, if True, extract calibrated probabilities
    """
    # collect dates and probability arrays
    dates = [x['date'] for x in valid_results]
    proba_key = 'calibrator_predictions_proba' if calibrator else 'predictions_proba'
    proba = np.vstack([x[proba_key] for x in valid_results])

    # build DataFrame
    proba_df = pd.DataFrame(
        data=proba,
        index=pd.DatetimeIndex(dates).tz_localize('UTC'),
        columns=sectors
    )
    return proba_df

def save_prediction_proba_df(proba_df, config_path, calibrator=False):
    """
    Save the predicted probabilities DataFrame to CSV using paths defined in config.
    calibrator: bool, if True, save as calibrated probabilities file
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))
    output_dir = os.path.join(base_dir, "../output")
    os.makedirs(output_dir, exist_ok=True)
    filename = 'calibrator_prediction_proba.csv' if calibrator else 'prediction_proba.csv'
    file_path = os.path.join(output_dir, filename)
    # Flush file before saving
    flush_file(file_path)
    proba_df.to_csv(file_path)
    return file_path

def analyze_top_sector_predictions(valid_results, sectors, returns_df):
    # Create empty DataFrames for probabilities and true labels
    dates = [pd.to_datetime(x['date']).tz_localize('UTC') for x in valid_results]
    proba_df = pd.DataFrame(index=pd.DatetimeIndex(dates), columns=sectors)
    true_df = pd.DataFrame(index=pd.DatetimeIndex(dates), columns=sectors)
    
    # Fill DataFrames with probabilities and true labels
    for i, result in enumerate(valid_results):
        proba_df.iloc[i] = result['predictions_proba']
        true_df.iloc[i] = result['true_labels']
    
    # For each date, find the sector with highest probability
    top_sectors = proba_df.idxmax(axis=1)
    top_probas = proba_df.max(axis=1)
    
    # Get the true labels for the top sectors
    top_true_labels = pd.Series(index=dates)
    for date in dates:
        top_sector = top_sectors[date]
        top_true_labels[date] = true_df.loc[date, top_sector]
    
    # Create final DataFrame
    final_df = pd.DataFrame({
        'top_sector': top_sectors,
        'predicted_proba': top_probas,
        'true_label': top_true_labels
    })
    
    # Add returns for top sectors
    final_df['actual_return'] = final_df.apply(lambda x: returns_df[x['top_sector']].loc[x.name], axis=1)
    
    return final_df

def compare_predictions(valid_results):
    comparison_results = []
    for result in valid_results:
        regular_pred = result['predictions']
        calibrated_pred = result['calibrator_predictions']
        date = result['date']
        
        # Find differences
        differences = regular_pred != calibrated_pred
        
        if np.any(differences):
            comparison_results.append({
                'date': date,
                'total_differences': np.sum(differences),
                'percent_different': (np.sum(differences) / len(differences)) * 100,
                'regular_positive': np.sum(regular_pred),
                'calibrated_positive': np.sum(calibrated_pred)
            })
    
    if comparison_results:
        return pd.DataFrame(comparison_results).set_index('date')
    else:
        return pd.DataFrame(columns=['total_differences', 'percent_different', 'regular_positive', 'calibrated_positive'])

def analyze_predictions_accuracy(valid_results, sectors=None, calibrated=False):
    # Convert predictions and true labels to arrays
    pred_key = 'calibrator_predictions' if calibrated else 'predictions'
    pred_array = np.array([x[pred_key] for x in valid_results])
    true_array = np.array([x['true_labels'] for x in valid_results])
    dates = np.array([x['date'] for x in valid_results])

    # Calculate results using vectorized operations
    res_1 = np.where(np.logical_and(pred_array,true_array), np.ones_like(pred_array),np.zeros_like(pred_array))
    res_2 = np.where(np.logical_and(pred_array,np.logical_not(true_array)), -np.ones_like(pred_array),res_1)

    # Create result DataFrame 
    result_df = pd.DataFrame(index=pd.DatetimeIndex(dates),
                           columns=sectors,
                           data=res_2)
    
    return result_df