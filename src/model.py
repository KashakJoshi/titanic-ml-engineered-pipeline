from sklearn.linear_model import LogisticRegression
from .config import LOGISTIC_PARAMS   # because inside same src package


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(**LOGISTIC_PARAMS)
    model.fit(X_train, y_train)
    return model






from sklearn.ensemble import RandomForestClassifier
from .config import RANDOM_FOREST_PARAMS


def train_random_forest(X_train, y_train):
    """
    Train Random Forest model.
    """
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)
    return model






from sklearn.ensemble import GradientBoostingClassifier
from .config import GRADIENT_BOOSTING_PARAMS


def train_gradient_boosting(X_train, y_train):
    """
    Train Gradient Boosting model.
    """
    model = GradientBoostingClassifier(**GRADIENT_BOOSTING_PARAMS)
    model.fit(X_train, y_train)
    return model







from catboost import CatBoostClassifier
from .config import CATBOOST_PARAMS


def train_catboost(X_train, y_train):
    """
    Train CatBoost model.
    """
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    return model






from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier


def build_catboost(params):
    return CatBoostClassifier(**params)


def build_logistic():
    return LogisticRegression(max_iter=1000)








def ensemble_predict(cat_model, log_model, X, weights):
    cat_pred = cat_model.predict_proba(X)[:, 1]
    log_pred = log_model.predict_proba(X)[:, 1]
    
    final_pred = (
        weights["catboost"] * cat_pred +
        weights["logistic"] * log_pred
    )
    
    return (final_pred > 0.5).astype(int)