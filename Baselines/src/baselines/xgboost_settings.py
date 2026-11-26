"""XGBoost settings used by the Rebuttal Figure 2 worker."""

LOCKED_XGBOOST = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "n_jobs": 1,
    "tree_method": "hist",
}
