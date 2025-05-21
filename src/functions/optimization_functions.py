import pandas as pd
import optuna
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def average_precision_eval(preds, data):
    # 1) Extract true labels as a 1-D numpy array
    y_true = data.get_label()                     # array-like (n_samples,)
    # 2) preds is already a 1-D array of probabilities (or raw scores)
    # 3) Compute AP with y_true first, then preds
    ap = average_precision_score(y_true, preds)   # sklearn signature: (y_true, y_score) :contentReference[oaicite:1]{index=1}
    # 4) Return name, value, and flag indicating “higher is better”
    return 'average_precision', ap, True

def average_precision_eval_sklearn(preds, dataset):
    # Extract true labels directly from the input dataset
    y_true = dataset.astype(int)
    # Compute average precision
    ap = average_precision_score(y_true, preds)
    # Return name, value, and flag indicating "higher is better"
    return 'average_precision', ap, True

def auc_feval(y_pred, dataset):
    y_true = dataset.get_label()
    # y_pred in LGB is raw score (not necessarily probabilities), 
    # but roc_auc_score will work on scores directly.
    auc = roc_auc_score(y_true, y_pred)
    # Return name, result, is_higher_better
    return 'custom_auc', auc, True

def custom_logloss(preds, train_data):
    labels = train_data.get_label()
    alpha, beta = 1.0, 10.0  # beta>1 penaliza FP con más fuerza
    p = 1/(1+np.exp(-preds))
    loss = -((1-labels)*beta*np.log(1-p) + labels*alpha*np.log(p))
    return 'custom_logloss', -float(np.mean(loss)), True

def purge_cv_folds(fold_generator, distance: int) -> list:
    """
    Select folds with a minimum distance between them to avoid data leakage
    
    Args:
        fold_generator: Generator or iterable of fold indices
        distance: Minimum number of folds between selected folds
        
    Returns:
        List of selected fold indices
    """
    folds = list(fold_generator)
    selected = []
    
    if folds:
        current = len(folds) - 1  # Start from last fold
        while current >= 0:
            selected.append(current)
            current -= distance

    final_folds = [folds[i] for i in sorted(selected)]
    final_folds.reverse()
            
    return final_folds

def parse_hyperparameter_space(trial, param_space: dict) -> dict:
    """
    Parse hyperparameter space from dictionary and create optuna trial suggestions
    
    Args:
        trial: Optuna trial object
        param_space: Dictionary with parameter names and their space definition
        
    Returns:
        Dictionary with parameter names and their values
    """
    params = {}
    for param_name, space in param_space.items():
        if len(space) == 1:
            params[param_name] = space[0]
        else:
            suggest_type = space[0]
            low, high = space[1:3]
            step = space[3] if len(space) > 3 else None
            
            if suggest_type == 'int':
                if step is None:
                    step = 1
                params[param_name] = trial.suggest_int(param_name, low, high, step)
            elif suggest_type == 'float':
                params[param_name] = trial.suggest_float(param_name, low, high, step)
            elif suggest_type == 'loguniform':
                params[param_name] = trial.suggest_loguniform(param_name, low, high)
            elif suggest_type == 'uniform':
                params[param_name] = trial.suggest_float(param_name, low, high)
            elif suggest_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, low)
            elif suggest_type == 'discrete_uniform':
                params[param_name] = trial.suggest_discrete_uniform(param_name, low, high, step)
    return params