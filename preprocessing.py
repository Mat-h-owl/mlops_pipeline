"""
preprocessing.py — Limpieza, imputación, codificación y split temporal.

Genera df_train.csv, df_test.csv, df_val.csv en data/processed/.
El split es temporal por la columna `partition`:
    - Train : p1 … p7
    - Test  : p6, p7   (solapado, para eval durante entrenamiento)
    - Val   : p8, p9   (holdout interno)
    - OOT   : p10      (simula al profesor — nunca se entrena)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
import json

warnings.filterwarnings("ignore")

# ── Columnas de metadata (se excluyen del modelo) ──────────────────────────
ID_COLS = ["partition", "key_value", "codunicocli", "p_fecinformacion",
           "fch_creacion", "tip_doc"]
TARGET_COL = "target"


def load_raw_data(path: str) -> pd.DataFrame:
    """Carga el CSV crudo desde `path`."""
    df = pd.read_csv(path)
    print(f"[preprocessing] Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} cols")
    return df


def _detect_column_roles(df: pd.DataFrame):
    """Clasifica columnas en categóricas, numéricas e ids."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ID_COLS and c != TARGET_COL]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ID_COLS and c != TARGET_COL]
    return cat_cols, num_cols


def impute_missing(df: pd.DataFrame, cat_cols: list, num_cols: list) -> pd.DataFrame:
    """
    Imputa nulos:
      - Numéricas → mediana
      - Categóricas → moda (valor más frecuente)
    """
    df = df.copy()
    for c in num_cols:
        if df[c].isna().any():
            median_val = df[c].median()
            df[c] = df[c].fillna(median_val)

    for c in cat_cols:
        if df[c].isna().any():
            mode_val = df[c].mode()[0] if len(df[c].mode()) > 0 else "UNKNOWN"
            df[c] = df[c].fillna(mode_val)

    return df


def encode_categoricals(df: pd.DataFrame, cat_cols: list,
                        encoders: dict = None, fit: bool = True):
    """
    Label-encoding de columnas categóricas.
    Si fit=True, ajusta y retorna los encoders.
    Si fit=False, usa los encoders provistos (modo inferencia).
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for c in cat_cols:
        if fit:
            le = LabelEncoder()
            # Agregar clase "UNKNOWN" para valores no vistos en inferencia
            classes = list(df[c].dropna().unique()) + ["__UNKNOWN__"]
            le.fit(classes)
            encoders[c] = le
        else:
            le = encoders.get(c)
            if le is None:
                continue

        # Mapear valores no vistos a "__UNKNOWN__"
        known = set(le.classes_)
        df[c] = df[c].apply(lambda x: x if x in known else "__UNKNOWN__")
        df[c] = le.transform(df[c])

    return df, encoders


def temporal_split(df: pd.DataFrame):
    """
    Split temporal por columna `partition`.
    Ordena las particiones alfabéticamente y asigna:
        - Train : todas excepto las últimas 3
        - Test  : penúltimas 2 de train (solapado para eval)
        - Val   : antepenúltima y penúltima partición final
        - OOT   : última partición
    Para 10 particiones (p1…p10):
        Train = p1-p7, Test = p6-p7, Val = p8-p9, OOT = p10
    """
    partitions = sorted(df["partition"].unique())
    n = len(partitions)
    print(f"[preprocessing] Particiones encontradas ({n}): {partitions}")

    if n < 4:
        raise ValueError(f"Se necesitan al menos 4 particiones, se encontraron {n}")

    oot_parts = [partitions[-1]]                     # p10
    val_parts = [partitions[-3], partitions[-2]]      # p8, p9
    train_parts = partitions[:-3]                     # p1-p7
    test_parts = partitions[-5:-3]                    # p6, p7 (solapado)

    # Asegurar que test tiene al menos 1 partición
    if len(test_parts) == 0:
        test_parts = train_parts[-2:]

    df_train = df[df["partition"].isin(train_parts)].copy()
    df_test  = df[df["partition"].isin(test_parts)].copy()
    df_val   = df[df["partition"].isin(val_parts)].copy()
    df_oot   = df[df["partition"].isin(oot_parts)].copy()

    print(f"[preprocessing] Train: {train_parts} → {len(df_train):,} filas")
    print(f"[preprocessing] Test:  {test_parts} → {len(df_test):,} filas")
    print(f"[preprocessing] Val:   {val_parts} → {len(df_val):,} filas")
    print(f"[preprocessing] OOT:   {oot_parts} → {len(df_oot):,} filas")

    return df_train, df_test, df_val, df_oot


def run_preprocessing(input_path: str,
                      output_dir: str = "data/processed",
                      save: bool = True):
    """
    Pipeline de preprocesamiento completo.

    Returns:
        df_train, df_test, df_val, meta (dict con encoders, cat_cols, num_cols)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Cargar datos
    df = load_raw_data(input_path)

    # 2. Detectar roles de columnas
    cat_cols, num_cols = _detect_column_roles(df)
    print(f"[preprocessing] Categóricas ({len(cat_cols)}): {cat_cols}")
    print(f"[preprocessing] Numéricas ({len(num_cols)}): {num_cols}")

    # 3. Imputar nulos
    df = impute_missing(df, cat_cols, num_cols)
    null_pct = df[cat_cols + num_cols].isna().mean().sum()
    print(f"[preprocessing] Nulos restantes en features: {null_pct:.4f}")

    # 4. Split temporal (antes de encoding para reportar particiones correctas)
    df_train, df_test, df_val, df_oot = temporal_split(df)

    # 5. Encoding categóricas — fit sobre train, transform sobre todo
    df_train, encoders = encode_categoricals(df_train, cat_cols, fit=True)
    df_test, _  = encode_categoricals(df_test,  cat_cols, encoders=encoders, fit=False)
    df_val, _   = encode_categoricals(df_val,   cat_cols, encoders=encoders, fit=False)
    df_oot, _   = encode_categoricals(df_oot,   cat_cols, encoders=encoders, fit=False)

    # 6. Guardar splits
    if save:
        df_train.to_csv(os.path.join(output_dir, "df_train.csv"), index=False)
        df_test.to_csv(os.path.join(output_dir,  "df_test.csv"),  index=False)
        df_val.to_csv(os.path.join(output_dir,   "df_val.csv"),   index=False)
        df_oot.to_csv(os.path.join(output_dir,   "df_oot.csv"),   index=False)
        print(f"[preprocessing] Archivos guardados en {output_dir}/")

        # Guardar metadata (nombres de columnas para referencia)
        meta_info = {
            "id_cols": ID_COLS,
            "target_col": TARGET_COL,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
        }
        with open(os.path.join(output_dir, "meta.json"), "w") as f:
            json.dump(meta_info, f, indent=2)

    meta = {
        "encoders": encoders,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }
    return df_train, df_test, df_val, meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/dataset.csv")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()
    run_preprocessing(args.input, args.output_dir)
