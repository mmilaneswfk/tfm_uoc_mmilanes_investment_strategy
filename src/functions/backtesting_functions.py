import lightgbm as lgb
import pandas as pd

from typing import List, Tuple, Dict, Optional, Union
import os
import pickle
import numpy as np


#--------------------------
# Data Preparation Functions
#--------------------------

def create_lgbm_dataset(X: pd.DataFrame, 
                        y: pd.Series, 
                        lookback_period: int, 
                        n_test_periods: int, 
                        end_date: Optional[str] = None,
                        sample_weights: Optional[pd.DataFrame] = None,
                        categorical_features: Optional[List[str]] = None

                        ) -> Tuple[lgb.Dataset, lgb.Dataset]:
    """
    Creates LightGBM datasets for training and validation based on date splitting.
    
    Args:

        X: Input features DataFrame with MultiIndex (first level security, second level datetime)
        y: Target variable Series with matching index to X
        lookback_period: Number of periods to look back for training
        n_test_periods: Number of periods to use for validation
        end_date: Filter data up to this date (format: 'YYYY-MM-DD')
        sample_weights: Optional weights for training/validation samples
        categorical_features: List of categorical feature names
    
    Returns:
        Tuple of (train_dataset, valid_dataset) LightGBM dataset objects
    """
    # Filter by end_date if provided, otherwise use the latest date in the dataset
    if end_date:
        end_date = pd.to_datetime(end_date)
    else:
        end_date = X.index.get_level_values('date').max()

    
    # Calculate start date
    start_date = end_date - pd.Timedelta(weeks=(lookback_period + n_test_periods))

    # Localize dates in UTC
    if not end_date.tz:
        end_date = end_date.tz_localize('UTC')
    start_date = start_date.tz_localize('UTC')
    
    # Filter DataFrame and Series to include only the desired date range

    mask = (X.index.get_level_values('date') >= start_date) & (X.index.get_level_values('date') <= end_date)

    X = X[mask]
    y = y[mask]
    
    # Get unique dates from second level of index
    dates = X.index.get_level_values('date').unique()
    
    # Calculate split indices
    if type(y) == pd.Series:
        test_idx = 1
    else:
        test_idx = y.shape[1]
    back_idx = len(dates) - test_idx
    split_idx = len(dates) - n_test_periods - test_idx
    start_idx = max(0, split_idx - lookback_period)
    
    # Split data into train and validation sets
    train_dates = dates[start_idx:split_idx]
    valid_dates = dates[split_idx:back_idx]
    test_dates = dates[back_idx:]
    
    # Filter DataFrame and Series by dates
    X_train = X.loc[pd.IndexSlice[:, train_dates],:]
    y_train = y.loc[pd.IndexSlice[:, train_dates]]
    X_valid = X.loc[pd.IndexSlice[:, valid_dates],:]
    y_valid = y.loc[pd.IndexSlice[:, valid_dates]]
    X_test = X.loc[pd.IndexSlice[:, test_dates],:]
    y_test = y.loc[pd.IndexSlice[:, test_dates]]

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
    
    return train_dataset, valid_dataset, X_test, y_test

#--------------------------
# Data Extraction Functions
#--------------------------

def extract_predictions(valid_results: List[Dict], 
                        sectors: List[str], 
                        calibrator: bool = False) -> pd.DataFrame:
    """
    Build a DataFrame of plain predictions for each sector/date.

    Args:
        valid_results: List of dicts returned in the backtest loop
        sectors: List of sector names in the same order as model inputs
        calibrator: If True, extract calibrated predictions

    Returns:
        DataFrame with plain predictions for each sector and date
    """
    # Determine the key for predictions based on whether calibrator is used
    pred_key = 'calibrator_predictions' if calibrator else 'predictions'

    # Collect dates and prediction arrays
    dates = [x['date'] for x in valid_results]
    predictions = np.vstack([x[pred_key] for x in valid_results])

    # Build DataFrame
    predictions_df = pd.DataFrame(
        data=predictions,
        index=pd.DatetimeIndex(dates).tz_localize('UTC'),
        columns=sectors
    )
    return predictions_df

def extract_predictions_proba(valid_results: List[Dict], 
                             sectors: List[str], 
                             calibrator: bool = False) -> pd.DataFrame:
    """
    Build a DataFrame of predicted probabilities for each sector/date.

    Args:
        valid_results: List of dicts returned in the backtest loop
        sectors: List of sector names in the same order as model inputs
        calibrator: If True, extract calibrated probabilities

    Returns:
        DataFrame with prediction probabilities for each sector and date
    """
    # Collect dates and probability arrays
    dates = [x['date'] for x in valid_results]
    proba_key = 'calibrator_predictions_proba' if calibrator else 'predictions_proba'
    proba = np.vstack([x[proba_key] for x in valid_results])

    # Build DataFrame
    proba_df = pd.DataFrame(
        data=proba,
        index=pd.DatetimeIndex(dates).tz_localize('UTC'),
        columns=sectors
    )
    return proba_df

def extract_true_labels(valid_results: List[Dict], sectors: List[str]) -> pd.DataFrame:
    """
    Build a DataFrame of true labels for each sector/date.

    Args:
        valid_results: List of dicts returned in the backtest loop
        sectors: List of sector names in the same order as model inputs

    Returns:
        DataFrame with true labels for each sector and date
    """
    # Collect dates and true label arrays
    dates = [x['date'] for x in valid_results]
    true_labels = np.vstack([x['true_labels'] for x in valid_results])

    # Build DataFrame
    true_labels_df = pd.DataFrame(
        data=true_labels,
        index=pd.DatetimeIndex(dates).tz_localize('UTC'),
        columns=sectors
    )
    return true_labels_df

#--------------------------
# Analysis Functions
#--------------------------

def analyze_predictions_accuracy(valid_results: List[Dict], 
                                sectors: Optional[List[str]] = None, 
                                calibrated: bool = False) -> pd.DataFrame:
    """
    Scores prediction accuracy: +1 for true positives, -1 for false positives, 0 otherwise.
    
    Args:
        valid_results: List of backtest results
        sectors: List of sector names for columns
        calibrated: Whether to use calibrated predictions
        
    Returns:
        DataFrame with scores by date and sector
    """
    # Get predictions and true values
    pred_key = 'calibrator_predictions' if calibrated else 'predictions'
    pred_array = np.array([x[pred_key] for x in valid_results], dtype=bool)
    true_array = np.array([x['true_labels'] for x in valid_results], dtype=bool)
    dates = np.array([x['date'] for x in valid_results])

    # Calculate scores: +1 for true positives, -1 for false positives
    tp_score = (pred_array & true_array) * 1
    fp_score = (pred_array & ~true_array) * -1
    scores = tp_score + fp_score

    # Create result DataFrame
    if sectors is None and pred_array.ndim > 1:
        num_cols = pred_array.shape[1]
        sectors = [f'col_{i}' for i in range(num_cols)]
        print("Warning: 'sectors' not provided, using generic column names.")
    elif sectors is not None and pred_array.ndim > 1 and len(sectors) != pred_array.shape[1]:
         raise ValueError(f"Number of sectors ({len(sectors)}) does not match number of prediction columns ({pred_array.shape[1]})")

    result_df = pd.DataFrame(
        data=scores,
        index=pd.DatetimeIndex(dates),
        columns=sectors
    )

    return result_df

def analyze_top_sector_predictions(valid_results: List[Dict], 
                                  sectors: List[str], 
                                  returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance of selecting the top predicted sector.
    
    Args:
        valid_results: List of backtest results
        sectors: List of sector names
        returns_df: DataFrame with actual returns for each sector
        
    Returns:
        DataFrame with top sector analysis results
    """
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

def compare_predictions(valid_results: List[Dict]) -> pd.DataFrame:
    """
    Compare original and calibrated predictions to identify differences.
    
    Args:
        valid_results: List of backtest results
        
    Returns:
        DataFrame with comparison metrics between original and calibrated predictions
    """
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

#--------------------------
# File Operations Functions
#--------------------------

def flush_file(file_path: str) -> None:
    """
    Remove a file if it exists to ensure clean write.
    
    Args:
        file_path: Path to the file to be removed
    """
    if os.path.exists(file_path):
        os.remove(file_path)

def save_model(model: object, config_path: str) -> str:
    """
    Save the trained model to a pickle file.
    
    Args:
        model: Trained model object
        config_path: Path to the configuration file
        
    Returns:
        Path where the model was saved
    """
    model_path = os.path.join(os.path.dirname(config_path), '../models/lgbm_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Flush file before saving
    flush_file(model_path)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model_path

def save_model_parameters(model: object, config_path: str) -> str:
    """
    Save model parameters to a CSV file.
    
    Args:
        model: Trained model object
        config_path: Path to the configuration file
        
    Returns:
        Path where the parameters were saved
    """
    model_params_df = pd.DataFrame.from_dict(model.get_params(), orient='index', columns=['value'])
    params_path = os.path.join(os.path.dirname(config_path), '../output/model_parameters.csv')
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    # Flush file before saving
    flush_file(params_path)
    model_params_df.to_csv(params_path)
    return params_path

def save_feature_importances(model: object, X: pd.DataFrame, config_path: str) -> str:
    """
    Save feature importances to a CSV file.
    
    Args:
        model: Trained model object
        X: Feature DataFrame
        config_path: Path to the configuration file
        
    Returns:
        Path where the feature importances were saved
    """
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

def save_categorical_info(categorical_features: List[str], valid_df_path: str) -> str:
    """
    Save categorical feature information to a CSV file.
    
    Args:
        categorical_features: List of categorical feature names
        valid_df_path: Base path for validation data files
        
    Returns:
        Path where the categorical information was saved
    """
    cat_info_path = valid_df_path + '_categorical.csv'
    # Flush file before saving
    flush_file(cat_info_path)
    # Convert list to DataFrame and save as CSV
    pd.DataFrame({'feature_name': categorical_features}).to_csv(cat_info_path, index=False)
    return cat_info_path

def save_valid_dfs(valid_df_saved: List, X: pd.DataFrame, config_path: str) -> str:
    """
    Save validation datasets to CSV files.
    
    Args:
        valid_df_saved: List of validation datasets
        X: Feature DataFrame
        config_path: Path to the configuration file
        
    Returns:
        Base path where validation data files were saved
    """
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

def save_prediction_accuracy(accuracy_df: pd.DataFrame, calibrated_accuracy_df: pd.DataFrame, config_path: str) -> Tuple[str, str]:
    """
    Save prediction accuracy DataFrames to CSV files.
    
    Args:
        accuracy_df: DataFrame with prediction accuracy for original predictions
        calibrated_accuracy_df: DataFrame with prediction accuracy for calibrated predictions
        config_path: Path to the configuration file
        
    Returns:
        Tuple of paths where the accuracy files were saved
    """

    accuracy_path = os.path.join(os.path.dirname(config_path), '../output/prediction_accuracy.csv')
    calibrated_accuracy_path = os.path.join(os.path.dirname(config_path), '../output/prediction_accuracy_calibrated.csv')
    os.makedirs(os.path.dirname(accuracy_path), exist_ok=True)
    # Flush files before saving
    flush_file(accuracy_path)
    flush_file(calibrated_accuracy_path)
    accuracy_df.to_csv(accuracy_path)
    calibrated_accuracy_df.to_csv(calibrated_accuracy_path)
    return accuracy_path, calibrated_accuracy_path

def save_predictions(predictions_df: pd.DataFrame, config_path: str, filename: str = "predictions.csv") -> str:
    """
    Save plain predictions DataFrame to CSV.
    
    Args:
        predictions_df: DataFrame with predictions
        config_path: Path to the configuration file
        filename: Name of the output file
        
    Returns:
        Path where the predictions were saved
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))
    output_dir = os.path.join(base_dir, "../output")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    # Flush file before saving
    flush_file(file_path)
    predictions_df.to_csv(file_path)
    return file_path

def save_prediction_proba_df(proba_df: pd.DataFrame, config_path: str, calibrator: bool = False) -> str:
    """
    Save predicted probabilities DataFrame to CSV.
    
    Args:
        proba_df: DataFrame with prediction probabilities
        config_path: Path to the configuration file
        calibrator: If True, save as calibrated probabilities file
        
    Returns:
        Path where the probabilities were saved
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

def save_top_sector_analysis(top_sector_analysis: pd.DataFrame, config_path: str) -> str:
    """
    Save top sector analysis results to CSV.
    
    Args:
        top_sector_analysis: DataFrame with top sector analysis
        config_path: Path to the configuration file
        
    Returns:
        Path where the analysis was saved
    """
    output_dir = os.path.join(os.path.dirname(config_path), '../output')
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, 'top_sector_analysis.csv')
    # Flush file before saving
    flush_file(file_path)
    top_sector_analysis.to_csv(file_path)
    return file_path


def save_gains_analysis(total_gains: pd.Series, potential_gains: pd.Series, config_path: str) -> str:
    """
    Save total and potential gains to CSV.
    
    Args:
        total_gains: Series with total gains
        potential_gains: Series with potential gains
        config_path: Path to the configuration file
        
    Returns:
        Path where the gains analysis was saved
    """
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

def save_correlation_data(correlation_data: pd.DataFrame, config_path: str) -> str:
    """
    Save correlation data to CSV.
    
    Args:
        correlation_data: DataFrame with correlation data
        config_path: Path to the configuration file
        
    Returns:
        Path where the correlation data was saved
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))
    output_dir = os.path.join(base_dir, "../output")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "correlation_data.csv")
    flush_file(file_path)
    correlation_data.to_csv(file_path)
    return file_path

def save_dataframe_to_csv(
    data: Union[pd.DataFrame, pd.Series, Dict[str, Union[pd.DataFrame, pd.Series]]], 
    config_path: str, 
    filename: Union[str, Dict[str, str]] = None,
    subdir: str = "output",
    index: bool = True
) -> Union[str, Dict[str, str]]:
    """
    Generic function to save DataFrame(s) or Series to CSV file(s).
    
    Args:
        data: DataFrame, Series, or dictionary of DataFrames/Series to save
        config_path: Path to the configuration file for base directory reference
        filename: Filename(s) to use (if None, derived from keys in the data dictionary)
        subdir: Subdirectory within the base directory to save files (default: "output")
        index: Whether to include index in the CSV (default: True)
        
    Returns:
        Path where the file was saved or dictionary of paths for multiple files
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))
    output_dir = os.path.join(base_dir, f"../{subdir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle single DataFrame case
    if isinstance(data, (pd.DataFrame, pd.Series)):
        if filename is None:
            raise ValueError("Filename must be provided for single DataFrame/Series")
        file_path = os.path.join(output_dir, filename)
        flush_file(file_path)
        data.to_csv(file_path, index=index)
        return file_path
    
    # Handle dictionary of DataFrames case
    elif isinstance(data, dict):
        result_paths = {}
        
        # If filename is a string, use it as a prefix
        prefix = filename if isinstance(filename, str) else ""
        
        for key, df in data.items():
            # Determine filename for this dataframe
            if isinstance(filename, dict) and key in filename:
                df_filename = filename[key]
            elif prefix:
                df_filename = f"{prefix}_{key}.csv"
            else:
                df_filename = f"{key}.csv"
                
            file_path = os.path.join(output_dir, df_filename)
            flush_file(file_path)
            df.to_csv(file_path, index=index)
            result_paths[key] = file_path
            
        return result_paths
    
    else:
        raise TypeError("Data must be a DataFrame, Series, or dictionary of DataFrames/Series")
