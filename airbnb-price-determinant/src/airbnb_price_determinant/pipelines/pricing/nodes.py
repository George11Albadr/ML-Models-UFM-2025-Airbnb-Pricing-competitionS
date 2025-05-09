"""
This is a boilerplate pipeline 'pricing'
generated using Kedro 0.19.12
"""
# src/airbnb_price_determinant/pipelines/pricing/nodes.py

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split, GridSearchCV
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, PolynomialFeatures
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.decomposition import PCA

from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    BaggingRegressor,
    StackingRegressor
)
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


def preprocess(train: pd.DataFrame, test: pd.DataFrame, params: dict):
    """Eliminar columnas, separar X/y y aplicar PCA si está configurado."""
    drops = params["preprocessing"]["drop_columns"]
    train_proc = train.drop(columns=drops)
    test_proc  = test.drop(columns=drops)

    X = train_proc.drop("realSum", axis=1)
    y = train_proc["realSum"]

    # Pipeline de preprocesamiento
    num_cols = params["preprocessing"]["num_features"]
    cat_cols = params["preprocessing"]["categorical_features"]
    numeric_pipeline = SKPipeline([
        ("imputer", SimpleImputer(strategy=params["preprocessing"]["imputer_strategy"])),
        ("scaler",  StandardScaler())
    ])
    categorical_pipeline = SKPipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ], remainder="passthrough")

    # Ajustar PCA si se especifica
    if "pca_n_components" in params["dimensionality_reduction"]:
        preprocessor = SKPipeline([
            ("pre", preprocessor),
            ("pca", PCA(n_components=params["dimensionality_reduction"]["pca_n_components"]))
        ])

    # Transformar
    X_proc = preprocessor.fit_transform(X)
    test_proc_features = preprocessor.transform(test_proc.drop(columns=["id"], errors="ignore"))

    return X_proc, y, test_proc_features


def split_data(X: np.ndarray, y: pd.Series, params: dict):
    return train_test_split(
        X, y,
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"],
        shuffle=True
    )


def _grid_search(model, param_grid: dict, X_tr, y_tr, cv: dict, fit_params: dict = None):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv["n_splits"],
        scoring=cv["scoring"],
        n_jobs=-1,
        refit=True
    )
    if fit_params:
        gs.fit(X_tr, y_tr, **fit_params)
    else:
        gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_


def train_xgb(X_train, y_train, X_val, y_val, params: dict):
    model = XGBRegressor(random_state=params["training"]["random_state"])
    best, best_params = _grid_search(
        model,
        params["xgb"]["param_grid"],
        X_train, y_train,
        params["cv"],
        fit_params={
            "eval_set": [(X_val, y_val)],
            "early_stopping_rounds": params["xgb"]["fit_params"]["early_stopping_rounds"],
            "verbose": False
        }
    )
    return best, best_params


def train_rf(X_train, y_train, _, __, params: dict):
    model = RandomForestRegressor(random_state=params["training"]["random_state"])
    best, best_params = _grid_search(
        model,
        params["rf"]["param_grid"],
        X_train, y_train,
        params["cv"]
    )
    return best, best_params


def train_gb(X_train, y_train, _, __, params: dict):
    model = GradientBoostingRegressor(random_state=params["training"]["random_state"])
    best, best_params = _grid_search(
        model,
        params["gb"]["param_grid"],
        X_train, y_train,
        params["cv"]
    )
    return best, best_params


def train_svr(X_train, y_train, _, __, params: dict):
    model = SVR()
    best, best_params = _grid_search(
        model,
        params["svr"]["param_grid"],
        X_train, y_train,
        params["cv"]
    )
    return best, best_params


def train_sgd(X_train, y_train, _, __, params: dict):
    model = SGDRegressor(random_state=params["training"]["random_state"])
    best, best_params = _grid_search(
        model,
        params["sgd"]["param_grid"],
        X_train, y_train,
        params["cv"]
    )
    return best, best_params


def train_knn(X_train, y_train, _, __, params: dict):
    model = KNeighborsRegressor()
    best, best_params = _grid_search(
        model,
        params["knn"]["param_grid"],
        X_train, y_train,
        params["cv"]
    )
    return best, best_params


def train_stacking(models: dict, params: dict):
    """
    models: dict con estimadores ya entrenados
    stacking usa XGBoost como final_estimator
    """
    estimators = [(name, mdl) for name, (mdl, _) in models.items() if name != "stacking"]
    final = XGBRegressor(random_state=params["training"]["random_state"])
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=final,
        n_jobs=-1,
        passthrough=False
    )
    stack.fit(models["X_train"], models["y_train"])
    return stack, {}


def evaluate(models: dict, X_val, y_val):
    """
    Devuelve un DataFrame con RMSE y R² para cada modelo
    """
    results = []
    for name, mdl in models.items():
        preds = mdl.predict(X_val)
        results.append({
            "model": name,
            "rmse": mean_squared_error(y_val, preds, squared=False),
            "r2":   r2_score(y_val, preds)
        })
    return pd.DataFrame(results)


def predict_submission(best_model, test_features, params: dict):
    preds = best_model.predict(test_features)
    return pd.DataFrame({
        "id":    test_features[:, 0] if test_features.ndim > 1 else np.arange(len(preds)),
        "realSum": preds
    })
