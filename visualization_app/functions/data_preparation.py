import pandas as pd
import pickle
import streamlit as st
import lightgbm as lgb
import os

@st.cache_data
def load_data(base_path=".."):
    valid = pd.read_csv(os.path.join(base_path, "output/prediction_accuracy.csv"), index_col=0, parse_dates=True)
    valid_calibrated = pd.read_csv(os.path.join(base_path, "output/prediction_accuracy_calibrated.csv"), index_col=0, parse_dates=True)
    predicted_proba = pd.read_csv(
        os.path.join(base_path, "output/prediction_proba.csv"),
        index_col=0,
        parse_dates=True
    )
    calibrator_predicted_proba = pd.read_csv(
        os.path.join(base_path, "output/calibrator_prediction_proba.csv"),
        index_col=0,
        parse_dates=True
    )
    data = pd.read_csv(os.path.join(base_path, "output/valid_df_data.csv"))
    labels = pd.read_csv(os.path.join(base_path, "output/valid_df_label.csv"))
    dtypes = pd.read_csv(os.path.join(base_path, "output/valid_df_dtypes.csv"))
    categorical_features = pd.read_csv(os.path.join(base_path, "output/valid_df_categorical.csv"))['feature_name'].tolist()
    top_sector_analysis = pd.read_csv(os.path.join(base_path, "output/top_sector_analysis.csv"), parse_dates=True, index_col=0)
    total_gains = pd.read_csv(os.path.join(base_path, "output/gains_analysis.csv"), parse_dates=True, index_col=0)['total_gains']
    returns_spy = pd.read_csv(os.path.join(base_path, "output/returns_spy.csv"), parse_dates=True, index_col=0)['return_1'].sort_index().shift(-1)
    target = pd.read_csv(os.path.join(base_path, "output/target.csv"), parse_dates=True, index_col=0)
    # Filter target to dates contained in valid index
    target = target.loc[target.index.isin(valid.index)]
    correlation_data = pd.read_csv(os.path.join(base_path, "output/correlation_data.csv"), index_col=0)
    # Apply dtypes to columns
    for idx, dtype in zip(dtypes.iloc[:,0], dtypes.iloc[:,1]):
        data[idx] = data[idx].astype(dtype)
    valid_df = lgb.Dataset(data=data, label=labels.iloc[:,0], categorical_feature=categorical_features)
    fi = pd.read_csv(os.path.join(base_path, "output/feature_importances.csv"))
    fi['importance'] = fi['importance'].astype(int)
    params = pd.read_csv(os.path.join(base_path, "output/model_parameters.csv"))

    DATA_STORE = os.path.join(base_path, "data/assets.h5")

    with pd.HDFStore(DATA_STORE) as store:
        raw_returns = (store['data_raw'].sort_index())
        raw_returns = raw_returns['return_1'].unstack(level=0).sort_index(axis=1)

    with pd.HDFStore(DATA_STORE) as store:
        raw_spy = (store['spy_raw'].sort_index())
        raw_spy = raw_spy['return_1']

    return (valid, valid_calibrated, fi, params, valid_df, 
            top_sector_analysis, total_gains, predicted_proba, 
            returns_spy, target, correlation_data, raw_returns, raw_spy,
            calibrator_predicted_proba)

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