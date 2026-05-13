"""
main.py — Orquestador principal del pipeline ML E2E.

Modos de ejecución:
    python main.py --mode train      # Entrena y guarda modelo
    python main.py --mode inference   # Carga modelo y corre con datos nuevos

Lógica de re-entrenamiento automático:
    Si PSI > 0.25 o los grupos TLV colapsan → re-entrena automáticamente.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import json

from preprocessing import run_preprocessing, ID_COLS, TARGET_COL, \
    load_raw_data, impute_missing, _detect_column_roles, encode_categoricals
from training import train_and_log, load_model, _xy
from monitoring import run_monitoring, compute_psi, psi_flag
from postprocessing import run_postprocessing, save_replica, validate_groups

# ── Rutas por defecto ──────────────────────────────────────────────────────
DEFAULT_INPUT   = "data/raw/dataset.csv"
OUTPUT_DIR      = "data/processed"
POST_PATH       = "data/postprocessed/output_tlv.csv"
MODEL_DIR       = "models"
MONITOR_DIR     = "data/monitoring"

# ── Umbrales ───────────────────────────────────────────────────────────────
PSI_RETRAIN_THRESHOLD = 0.25
MIN_GROUPS_THRESHOLD  = 5


def mode_train(input_path: str, n_trials: int = 30):
    """
    Pipeline de entrenamiento completo:
    1. Preprocesamiento
    2. Entrenamiento (Optuna + MLflow)
    3. Monitoreo
    4. Postprocesamiento + réplica
    """
    print("=" * 70)
    print("  PIPELINE — MODO ENTRENAMIENTO")
    print("=" * 70)

    # ── 1. Preprocesamiento ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  ETAPA 1: Preprocesamiento")
    print("─" * 70)
    df_train, df_test, df_val, meta = run_preprocessing(input_path, OUTPUT_DIR)

    # ── 2. Entrenamiento ───────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  ETAPA 2: Entrenamiento (Optuna + MLflow)")
    print("─" * 70)
    run_id, model = train_and_log(
        train_path=os.path.join(OUTPUT_DIR, "df_train.csv"),
        test_path=os.path.join(OUTPUT_DIR,  "df_test.csv"),
        val_path=os.path.join(OUTPUT_DIR,   "df_val.csv"),
        n_trials=n_trials,
        model_dir=MODEL_DIR,
    )

    # ── 3. Monitoreo ──────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  ETAPA 3: Monitoreo (PSI + Métricas)")
    print("─" * 70)
    X_val, y_val = _xy(df_val)
    val_scores = model.predict_proba(X_val)[:, 1]
    monitoring_result = run_monitoring(
        df_train, df_val, val_scores,
        model=model, output_dir=MONITOR_DIR,
    )

    # ── 4. Postprocesamiento ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  ETAPA 4: Postprocesamiento (TLV + Grupos)")
    print("─" * 70)
    df_resultado, groups_ok = run_postprocessing(val_scores, df_val, POST_PATH)

    # Guardar réplica
    partition_label = df_val["partition"].mode().iloc[0] if "partition" in df_val.columns else "train"
    save_replica(df_resultado, table="EC_OMNICANAL", partition=partition_label)

    print("\n" + "=" * 70)
    print("  PIPELINE ENTRENAMIENTO — COMPLETADO")
    print(f"  Run ID: {run_id}")
    print(f"  PSI: {monitoring_result['psi_score']:.4f} ({monitoring_result['psi_flag']})")
    print(f"  AUC val: {monitoring_result['auc_val']:.4f}")
    print(f"  Grupos OK: {groups_ok}")
    print("=" * 70)


def mode_inference(input_path: str, n_trials: int = 30):
    """
    Pipeline de inferencia con datos nuevos:
    1. Preprocesamiento de datos nuevos
    2. Carga modelo existente
    3. Monitoreo (PSI) → re-entrena si es necesario
    4. Postprocesamiento + réplica
    """
    print("=" * 70)
    print("  PIPELINE — MODO INFERENCIA")
    print("=" * 70)

    # ── Cargar modelo entrenado ────────────────────────────────────────────
    model = load_model(MODEL_DIR)

    # ── Cargar datos de referencia (train) para PSI ────────────────────────
    train_path = os.path.join(OUTPUT_DIR, "df_train.csv")
    if not os.path.exists(train_path):
        print("[main] ⚠️ No se encontró df_train.csv. Ejecute primero --mode train")
        return
    df_train = pd.read_csv(train_path)

    # ── Preprocesar datos nuevos ──────────────────────────────────────────
    print("\n[main] Preprocesando datos de inferencia...")
    df_new = load_raw_data(input_path)
    cat_cols, num_cols = _detect_column_roles(df_new)
    df_new = impute_missing(df_new, cat_cols, num_cols)

    # Cargar encoders desde meta
    meta_path = os.path.join(OUTPUT_DIR, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta_info = json.load(f)
        cat_cols_saved = meta_info.get("cat_cols", cat_cols)
    else:
        cat_cols_saved = cat_cols

    # Encoding: cargar encoders del modelo (simplificado - usar LabelEncoder nuevos)
    from sklearn.preprocessing import LabelEncoder
    # Para inferencia, hacemos encoding simple
    for c in cat_cols_saved:
        if c in df_new.columns and df_new[c].dtype == "object":
            le = LabelEncoder()
            df_new[c] = le.fit_transform(df_new[c].astype(str))

    # ── Scoring ───────────────────────────────────────────────────────────
    X_new, y_new = _xy(df_new)

    # Alinear columnas con las del modelo
    feature_names_path = os.path.join(MODEL_DIR, "feature_names.json")
    if os.path.exists(feature_names_path):
        with open(feature_names_path) as f:
            expected_features = json.load(f)
        # Agregar columnas faltantes con 0
        for c in expected_features:
            if c not in X_new.columns:
                X_new[c] = 0
        X_new = X_new[expected_features]

    new_scores = model.predict_proba(X_new)[:, 1]

    # ── Monitoreo PSI ─────────────────────────────────────────────────────
    print("\n[main] Calculando PSI...")
    X_train_ref, _ = _xy(df_train)
    train_scores = model.predict_proba(X_train_ref)[:, 1]

    monitoring_result = run_monitoring(
        df_train, df_new, new_scores,
        train_scores=train_scores,
        output_dir=MONITOR_DIR,
    )

    # ── Postprocesamiento ─────────────────────────────────────────────────
    print("\n[main] Postprocesamiento...")
    df_resultado, groups_ok = run_postprocessing(new_scores, df_new, POST_PATH)

    partition_label = "inference"
    if "partition" in df_new.columns:
        partition_label = df_new["partition"].mode().iloc[0]
    save_replica(df_resultado, table="EC_OMNICANAL", partition=partition_label)

    # ── Verificar si se necesita re-entrenamiento ─────────────────────────
    needs_retrain = (
        monitoring_result["psi_flag"] == "ALERT" or
        not groups_ok
    )

    if needs_retrain:
        print("\n" + "!" * 70)
        print("  ⚠️  RE-ENTRENAMIENTO AUTOMÁTICO ACTIVADO")
        reason = []
        if monitoring_result["psi_flag"] == "ALERT":
            reason.append(f"PSI={monitoring_result['psi_score']:.4f} > {PSI_RETRAIN_THRESHOLD}")
        if not groups_ok:
            reason.append("Grupos TLV colapsaron")
        print(f"  Razón: {', '.join(reason)}")
        print("!" * 70)

        # Re-entrenar con los datos nuevos incorporados
        mode_train(input_path, n_trials=n_trials)
    else:
        print("\n" + "=" * 70)
        print("  PIPELINE INFERENCIA — COMPLETADO")
        print(f"  PSI: {monitoring_result['psi_score']:.4f} ({monitoring_result['psi_flag']})")
        print(f"  AUC: {monitoring_result['auc_val']:.4f}")
        print(f"  Grupos OK: {groups_ok}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ML E2E — cu_venta (XGBoost + Optuna + MLflow)"
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["train", "inference"],
        help="Modo de ejecución: 'train' o 'inference'"
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT,
        help="Ruta al dataset CSV de entrada"
    )
    parser.add_argument(
        "--n_trials", type=int, default=30,
        help="Número de trials de Optuna (default: 30)"
    )
    args = parser.parse_args()

    if args.mode == "train":
        mode_train(args.input, n_trials=args.n_trials)
    elif args.mode == "inference":
        mode_inference(args.input, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
