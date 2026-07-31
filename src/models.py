"""Model training: CLV regression (tuned XGBoost) + churn classification."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from .features import FEATURE_COLS

RANDOM_STATE = 42

REG_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}


def split_data(features: pd.DataFrame):
    X = features[FEATURE_COLS]
    y_reg = features["actual_clv_holdout"]
    y_clf = features["churned"]
    return train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
    )


def train_regressor(X_train, y_reg_train, n_iter: int = 25):
    """Hyperparameter-tuned XGBoost regressor for CLV (reuses Lesson 4's
    systematic-search approach instead of hand-picked params)."""
    base_model = xgb.XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror")
    search = RandomizedSearchCV(
        base_model, param_distributions=REG_PARAM_DIST, n_iter=n_iter, cv=3,
        scoring="neg_mean_absolute_error", random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_reg_train)
    return search.best_estimator_, search.best_params_


def evaluate_regressor(model, X_test, y_test, y_train):
    y_pred = model.predict(X_test)
    baseline_pred = [y_train.mean()] * len(y_test)
    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "baseline_mae": float(mean_absolute_error(y_test, baseline_pred)),
    }


def train_best_classifier(X_train, y_clf_train, X_test, y_clf_test):
    """Train Logistic Regression / Decision Tree / Random Forest (Lesson 3
    reuse) and return the one with the best ROC-AUC."""
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE),
    }

    results = []
    fitted = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_clf_train)
        proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_clf_test, proba)
        results.append({"model": name, "roc_auc": float(auc)})
        fitted[name] = clf

    results.sort(key=lambda r: r["roc_auc"], reverse=True)
    best_name = results[0]["model"]
    return fitted[best_name], best_name, results
