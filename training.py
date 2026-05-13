"""
training.py — Entrenamiento XGBoost con Optuna (HPO) + registro en MLflow.

Busca hiperparámetros óptimos con Optuna (30 trials por defecto),
re-entrena el modelo final con los mejores parámetros y registra
todo en MLflow (parámetros, métricas, modelo serializado).
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
import joblib
from sklearn.metrics import roc_auc_score
import warnings
import json

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Columnas a excluir del entrenamiento ───────────────────────────────────
ID_COLS = ["partition", "key_value", "codunicocli", "p_fecinformacion",
           "fch_creacion", "tip_doc"]
TARGET_COL = "target"


def _xy(df: pd.DataFrame):
    """Separa features (X) y target (y), eliminando columnas de metadata."""
    drop = [c for c in ID_COLS + [TARGET_COL] if c in df.columns]
    X = df.drop(columns=drop)
    y = df[TARGET_COL]
    return X, y


def train_and_log(train_path: str, test_path: str, val_path: str,
                  n_trials: int = 30,
                  experiment_name: str = "cu_venta_e2e",
                  model_dir: str = "models"):
    """
    Busca hiperparámetros con Optuna y registra el mejor modelo en MLflow.

    Args:
        train_path: Ruta a df_train.csv
        test_path:  Ruta a df_test.csv
        val_path:   Ruta a df_val.csv
        n_trials:   Cantidad de trials de Optuna (default 30)
        experiment_name: Nombre del experimento en MLflow
        model_dir:  Directorio donde guardar el modelo .joblib

    Returns:
        run_id (str), modelo entrenado (XGBClassifier)
    """
    os.makedirs(model_dir, exist_ok=True)

    # ── Cargar datos ───────────────────────────────────────────────────────
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    df_val   = pd.read_csv(val_path)

    X_train, y_train = _xy(df_train)
    X_test,  y_test  = _xy(df_test)
    X_val,   y_val   = _xy(df_val)

    feature_names = list(X_train.columns)
    print(f"[training] Features: {len(feature_names)}")
    print(f"[training] Train: {X_train.shape}, Test: {X_test.shape}, Val: {X_val.shape}")
    print(f"[training] Target balance (train): {y_train.mean():.4f}")

    # ── Optuna HPO ─────────────────────────────────────────────────────────
    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 50, 500),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":           trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_pred_proba)

    print(f"[training] Iniciando búsqueda Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"[training] Mejor AUC (test): {study.best_value:.4f}")
    print(f"[training] Mejores parámetros: {study.best_params}")

    # ── Re-entrenar con mejores parámetros ────────────────────────────────
    best_params = {
        **study.best_params,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    # ── Métricas sobre val ─────────────────────────────────────────────────
    auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    auc_val  = roc_auc_score(y_val,  model.predict_proba(X_val)[:, 1])
    print(f"[training] AUC test: {auc_test:.4f}  |  AUC val: {auc_val:.4f}")

    # ── MLflow ─────────────────────────────────────────────────────────────
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("test_auc", auc_test)
        mlflow.log_metric("val_auc", auc_val)
        mlflow.log_metric("optuna_best_auc", study.best_value)
        mlflow.log_metric("n_trials", n_trials)
        mlflow.xgboost.log_model(model, "model",
                                 registered_model_name="cu_venta_xgb")
        run_id = run.info.run_id
        print(f"[training] MLflow run_id: {run_id}")

    # ── Guardar modelo localmente ──────────────────────────────────────────
    model_path = os.path.join(model_dir, "best_model.joblib")
    joblib.dump(model, model_path)

    # Guardar feature names
    with open(os.path.join(model_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Guardar mejores parámetros
    with open(os.path.join(model_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"[training] Modelo guardado en {model_path}")
    return run_id, model


def load_model(model_dir: str = "models"):
    """Carga el modelo entrenado desde disco."""
    model_path = os.path.join(model_dir, "best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró modelo en {model_path}")
    model = joblib.load(model_path)
    print(f"[training] Modelo cargado desde {model_path}")
    return model


if __name__ == "__main__":
    train_and_log(
        train_path="data/processed/df_train.csv",
        test_path="data/processed/df_test.csv",
        val_path="data/processed/df_val.csv",
    )
