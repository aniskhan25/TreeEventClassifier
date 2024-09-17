import shap
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_importance(X, y, cutoff=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(xgb_model)

    shap_values = explainer.shap_values(X_test)

    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.Series(shap_importance, index=X_test.columns)

    important_features = shap_importance_df[shap_importance_df > cutoff].index.tolist()
    feature_weights = shap_importance_df[important_features].tolist()

    logger.info(f"Important features with SHAP importance > 0.1: {important_features}")
    logger.info(f"Important features weights: {feature_weights}")

    return important_features, feature_weights

'''
Note: Additional requirement for XGBoost

brew install libomp
'''
