import pandas as pd
import pickle
import streamlit as st
import lightgbm as lgb
import os

@st.cache_data
def load_data(base_path=".."):
    # Load prediction accuracy data
    valid = pd.read_csv(os.path.join(base_path, "output/prediction_accuracy.csv"), 
                       index_col=0, parse_dates=True)
    valid_calibrated = pd.read_csv(os.path.join(base_path, "output/prediction_accuracy_calibrated.csv"), 
                                  index_col=0, parse_dates=True)
    
    # Load prediction probabilities
    predicted_proba = pd.read_csv(os.path.join(base_path, "output/prediction_proba.csv"),
                                 index_col=0, parse_dates=True)
    calibrator_predicted_proba = pd.read_csv(os.path.join(base_path, "output/calibrator_prediction_proba.csv"),
                                           index_col=0, parse_dates=True)
    
    # Load model predictions
    predictions = pd.read_csv(os.path.join(base_path, "output/model_predictions.csv"), 
                             index_col=0, parse_dates=True)
    calibrator_predictions = pd.read_csv(os.path.join(base_path, "output/calibrator_predictions.csv"), 
                                        index_col=0, parse_dates=True)
    
    # Load true labels data
    true_labels = pd.read_csv(os.path.join(base_path, "output/true_labels.csv"), 
                            index_col=0, parse_dates=True)
    
    # Load dataset components
    data = pd.read_csv(os.path.join(base_path, "output/valid_df_data.csv"))
    labels = pd.read_csv(os.path.join(base_path, "output/valid_df_label.csv"))
    dtypes = pd.read_csv(os.path.join(base_path, "output/valid_df_dtypes.csv"))
    categorical_features = pd.read_csv(os.path.join(base_path, "output/valid_df_categorical.csv"))['feature_name'].tolist()
    
    # Apply dtypes to columns
    for idx, dtype in zip(dtypes.iloc[:,0], dtypes.iloc[:,1]):
        data[idx] = data[idx].astype(dtype)
    
    # Create LightGBM dataset
    valid_df = lgb.Dataset(data=data, 
                          label=labels.iloc[:,0], 
                          categorical_feature=categorical_features)
    
    # Load target and correlation data
    target = pd.read_csv(os.path.join(base_path, "output/target.csv"), 
                        parse_dates=True, index_col=0)
    # Filter target to dates contained in valid index
    target = target.loc[target.index.isin(valid.index)]
    
    correlation_data = pd.read_csv(os.path.join(base_path, "output/correlation_data.csv"), 
                                   index_col=0)
    
    # Load feature importances and model parameters
    fi = pd.read_csv(os.path.join(base_path, "output/feature_importances.csv"))
    fi['importance'] = fi['importance'].astype(int)
    
    params = pd.read_csv(os.path.join(base_path, "output/model_parameters.csv"))
    
    # Load raw data from HDF store
    DATA_STORE = os.path.join(base_path, "data/assets.h5")
    
    with pd.HDFStore(DATA_STORE) as store:
        raw_returns = store['data_raw'].sort_index()
        raw_returns = raw_returns['return_1'].unstack(level=0).sort_index(axis=1)
    
    with pd.HDFStore(DATA_STORE) as store:
        raw_spy = store['spy_raw'].sort_index()
        raw_spy = raw_spy['return_1']
    
    return (valid, valid_calibrated, fi, params, valid_df, 
            predicted_proba, predictions, calibrator_predictions,
            target, correlation_data, raw_returns, raw_spy,
            calibrator_predicted_proba, true_labels)

@st.cache_resource
def load_model(base_path=".."):
    with open(os.path.join(base_path, "models/lgbm_model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model

def compute_precision(valid_df):
    tp = (valid_df == 1).sum(axis=1)
    fp = (valid_df == -1).sum(axis=1)
    denom = tp + fp
    precision = tp / denom
    # set precision to NaN where there are no positives or negatives
    precision = precision.mask(denom == 0)
    return pd.DataFrame({'TP': tp, 'FP': fp, 'Precision': precision})