"""
monitoring.py — PSI, AUC y Recall por decil (monitoreo de deriva).

Umbrales PSI:
    < 0.10   → OK   (sin deriva)
    0.10–0.25 → WARN (deriva moderada)
    > 0.25   → ALERT (deriva severa — dispara re-entrenamiento)
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score
import json
import warnings

warnings.filterwarnings("ignore")

ID_COLS = ["partition", "key_value", "codunicocli", "p_fecinformacion",
           "fch_creacion", "tip_doc"]
TARGET_COL = "target"


def psi_flag(psi: float) -> str:
    """Retorna la etiqueta de alerta según el valor de PSI."""
    if psi < 0.10:
        return "OK"
    elif psi < 0.25:
        return "WARN"
    return "ALERT"


def compute_psi(expected: np.ndarray, actual: np.ndarray,
                n_bins: int = 10, eps: float = 1e-4) -> float:
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones
    de scores, divididas en `n_bins` buckets (deciles).

    PSI = Σ (actual_i - expected_i) × ln(actual_i / expected_i)

    Args:
        expected: array de scores de referencia (train)
        actual:   array de scores nuevos (val/OOT)
        n_bins:   número de buckets (default 10 = deciles)
        eps:      suavizado para evitar log(0)

    Returns:
        PSI (float)
    """
    # Crear bins basados en los percentiles de expected
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] = np.inf

    # Contar proporciones en cada bin
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    expected_pct = expected_counts / len(expected) + eps
    actual_pct   = actual_counts   / len(actual)   + eps

    # PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_recall_by_decile(y_true: np.ndarray, scores: np.ndarray,
                             n_deciles: int = 10) -> pd.DataFrame:
    """
    Calcula el Recall acumulado por decil de score.
    Decil 1 = mayor score (mejores prospectos).

    Returns:
        DataFrame con columnas: decil, n_total, n_positivos, recall_acum, pct_poblacion
    """
    df = pd.DataFrame({"target": y_true, "score": scores})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Asignar deciles (1 = top, 10 = bottom)
    df["decil"] = pd.qcut(df.index, n_deciles, labels=False, duplicates="drop") + 1

    total_pos = df["target"].sum()
    rows = []
    cum_pos = 0

    for d in sorted(df["decil"].unique()):
        mask = df["decil"] == d
        n_total = mask.sum()
        n_pos   = df.loc[mask, "target"].sum()
        cum_pos += n_pos
        recall_acum = cum_pos / total_pos if total_pos > 0 else 0.0

        rows.append({
            "decil": d,
            "n_total": int(n_total),
            "n_positivos": int(n_pos),
            "tasa_conversion": round(n_pos / n_total, 4) if n_total > 0 else 0,
            "recall_acum": round(recall_acum, 4),
            "pct_poblacion": round(d / n_deciles, 2),
        })

    result = pd.DataFrame(rows)
    return result


def run_monitoring(df_train: pd.DataFrame, df_val: pd.DataFrame,
                   val_scores: np.ndarray,
                   train_scores: np.ndarray = None,
                   model=None,
                   output_dir: str = "data/monitoring"):
    """
    Calcula PSI sobre deciles de score, AUC y Recall en validación.

    Args:
        df_train:     DataFrame de entrenamiento
        df_val:       DataFrame de validación
        val_scores:   Scores de probabilidad sobre df_val
        train_scores: Scores de probabilidad sobre df_train (si None, se calculan)
        model:        Modelo entrenado (necesario si train_scores is None)
        output_dir:   Directorio de salida para reportes

    Returns:
        dict con psi_score, psi_flag, auc_val, recall_table, needs_retrain
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Calcular train_scores si no se proveen ─────────────────────────────
    if train_scores is None:
        if model is None:
            raise ValueError("Se necesita model o train_scores")
        drop = [c for c in ID_COLS + [TARGET_COL] if c in df_train.columns]
        X_train = df_train.drop(columns=drop)
        train_scores = model.predict_proba(X_train)[:, 1]

    # ── PSI ────────────────────────────────────────────────────────────────
    psi_val = compute_psi(train_scores, val_scores)
    flag = psi_flag(psi_val)
    print(f"[monitoring] PSI: {psi_val:.4f} → {flag}")

    # ── AUC ────────────────────────────────────────────────────────────────
    y_val = df_val[TARGET_COL].values
    auc_val = roc_auc_score(y_val, val_scores)
    print(f"[monitoring] AUC (val): {auc_val:.4f}")

    # ── Recall por decil ───────────────────────────────────────────────────
    recall_table = compute_recall_by_decile(y_val, val_scores)
    print(f"[monitoring] Recall por decil:")
    print(recall_table.to_string(index=False))

    # ── Guardar reporte ────────────────────────────────────────────────────
    report = {
        "psi_score": round(psi_val, 6),
        "psi_flag": flag,
        "auc_val": round(auc_val, 4),
        "needs_retrain": flag == "ALERT",
    }
    with open(os.path.join(output_dir, "monitoring_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    recall_table.to_csv(os.path.join(output_dir, "recall_by_decile.csv"), index=False)
    print(f"[monitoring] Reporte guardado en {output_dir}/")

    return {
        **report,
        "recall_table": recall_table,
    }


if __name__ == "__main__":
    print("[monitoring] Ejecutar desde main.py")
