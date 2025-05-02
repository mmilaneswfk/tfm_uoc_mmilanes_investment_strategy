import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import shap
from src.functions.BorutaShap_fix import BorutaShap

class BorutaShap2(BorutaPy):
    def __init__(self, estimator=None, n_estimators=1000, perc=100, alpha=0.05, 
                 two_step=True, max_iter=100, random_state=None, verbose=0):
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=n_estimators)
        super().__init__(estimator=estimator, n_estimators=n_estimators, 
                        perc=perc, alpha=alpha, two_step=two_step, 
                        max_iter=max_iter, random_state=random_state, 
                        verbose=verbose)

    def _calc_imp(self, X):
        explainer = shap.TreeExplainer(self.estimator)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return np.abs(shap_values).mean(axis=0)