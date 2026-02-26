# Logistic Regression Parameters

LOGISTIC_PARAMS = {
    "max_iter": 1000,
    "random_state": 42
}




# Random Forest Parameters

RANDOM_FOREST_PARAMS = {
    "n_estimators": 150,
    "max_depth": 6,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": 42
}





# GRADIENT BOOSTING PARAMETERS

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "max_depth": 3,
    "random_state": 42
}






# CATBOOST PARAMETERS

CATBOOST_PARAMS = {
    "depth": 6,
    "learning_rate": 0.10,
    "iterations": 300,
    "l2_leaf_reg": 5,
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
    "random_state": 42,
    "verbose": False
}

ENSEMBLE_WEIGHTS = {
    "catboost": 0.75,
    "logistic": 0.25
}