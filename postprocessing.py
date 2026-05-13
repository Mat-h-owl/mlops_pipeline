"""
postprocessing.py — Scoring TLV compuesto y segmentación en 10 grupos.

Calcula la puntuación compuesta TLV (Time-Life Value) y segmenta
la población en 10 grupos de ejecución usando la función get_groups()
del curso, SIN modificaciones a la fórmula ni a los cuantiles DIST_GE.
"""

import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

ID_COLS = ["partition", "key_value", "codunicocli", "p_fecinformacion",
           "fch_creacion", "tip_doc"]
TARGET_COL = "target"


def compute_tlv_score(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """
    Calcula el score compuesto TLV.

    TLV = score_modelo × (monto normalizado) × factor_tea

    El score TLV combina la probabilidad de compra (score del modelo)
    con el valor monetario de la oportunidad (monto) y el costo/rentabilidad
    (TEA = tasa efectiva anual).

    Args:
        df:     DataFrame con columnas 'monto', 'tea', etc.
        scores: Array de probabilidades del modelo

    Returns:
        DataFrame con columna 'score_tlv' agregada
    """
    df = df.copy()
    df["prob_score"] = scores

    # Normalizar monto al rango [0, 1]
    monto_min = df["monto"].min()
    monto_max = df["monto"].max()
    if monto_max > monto_min:
        df["monto_norm"] = (df["monto"] - monto_min) / (monto_max - monto_min)
    else:
        df["monto_norm"] = 1.0

    # Factor TEA: mayor TEA → mayor rentabilidad para el banco → mayor score
    tea_min = df["tea"].min()
    tea_max = df["tea"].max()
    if tea_max > tea_min:
        df["tea_norm"] = (df["tea"] - tea_min) / (tea_max - tea_min)
    else:
        df["tea_norm"] = 1.0

    # Score TLV compuesto
    df["score_tlv"] = (
        0.50 * df["prob_score"] +
        0.30 * df["monto_norm"] +
        0.20 * df["tea_norm"]
    )

    return df


def get_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segmenta la población en 10 grupos de ejecución (grupo_ejec_tlv)
    usando cuantiles DIST_GE sobre el score TLV.

    ⚠️ Esta función replica EXACTAMENTE la lógica del curso.
       No modificar la fórmula TLV ni los cuantiles DIST_GE.

    Grupo 1 = mejores prospectos (mayor score TLV)
    Grupo 10 = peores prospectos (menor score TLV)

    Args:
        df: DataFrame con columna 'score_tlv'

    Returns:
        DataFrame con columna 'grupo_ejec_tlv' (1–10)
    """
    df = df.copy()

    # Cuantiles DIST_GE: dividir en 10 grupos equidistantes por score_tlv
    try:
        df["grupo_ejec_tlv"] = pd.qcut(
            df["score_tlv"],
            q=10,
            labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # 1=mejor, 10=peor
            duplicates="drop",
        )
    except ValueError:
        # Si hay demasiados valores repetidos para qcut, usar ranking
        df["rank_tlv"] = df["score_tlv"].rank(method="first", ascending=False)
        n = len(df)
        df["grupo_ejec_tlv"] = pd.cut(
            df["rank_tlv"],
            bins=10,
            labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        df = df.drop(columns=["rank_tlv"])

    df["grupo_ejec_tlv"] = df["grupo_ejec_tlv"].astype(int)

    return df


def validate_groups(df: pd.DataFrame) -> bool:
    """
    Verifica que los grupos TLV no hayan colapsado.
    Si hay menos de 5 grupos distintos, se considera colapso → re-entrenar.
    """
    n_groups = df["grupo_ejec_tlv"].nunique()
    if n_groups < 5:
        print(f"[postprocessing] ⚠️ ALERTA: Solo {n_groups} grupos distintos (colapso)")
        return False
    print(f"[postprocessing] ✓ {n_groups} grupos distintos — OK")
    return True


def run_postprocessing(scores: np.ndarray, df: pd.DataFrame,
                       output_path: str = "data/postprocessed/output_tlv.csv"):
    """
    Pipeline de postprocesamiento completo.

    Args:
        scores:      Array de probabilidades del modelo
        df:          DataFrame original (con monto, tea, etc.)
        output_path: Ruta de salida para el CSV con TLV

    Returns:
        df_resultado: DataFrame con score_tlv y grupo_ejec_tlv
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Calcular score TLV
    df_resultado = compute_tlv_score(df, scores)
    print(f"[postprocessing] Score TLV — min: {df_resultado['score_tlv'].min():.4f}, "
          f"max: {df_resultado['score_tlv'].max():.4f}, "
          f"mean: {df_resultado['score_tlv'].mean():.4f}")

    # 2. Segmentar en grupos
    df_resultado = get_groups(df_resultado)

    # 3. Resumen por grupo
    resumen = (
        df_resultado
        .groupby("grupo_ejec_tlv")
        .agg(
            n_clientes=("score_tlv", "size"),
            score_tlv_mean=("score_tlv", "mean"),
            monto_mean=("monto", "mean"),
            tea_mean=("tea", "mean"),
            tasa_conversion=(TARGET_COL, "mean"),
        )
        .round(4)
    )
    print(f"\n[postprocessing] Resumen por grupo:")
    print(resumen.to_string())

    # 4. Validar que los grupos no colapsaron
    groups_ok = validate_groups(df_resultado)

    # 5. Guardar resultado
    cols_output = [c for c in df_resultado.columns
                   if c not in ["monto_norm", "tea_norm"]]
    df_resultado[cols_output].to_csv(output_path, index=False)
    print(f"\n[postprocessing] Resultado guardado en {output_path}")

    return df_resultado, groups_ok


def save_replica(df: pd.DataFrame, table: str = "EC_OMNICANAL",
                 partition: str = "202412",
                 output_dir: str = "data/replica"):
    """
    Guarda el resultado en formato pipe-delimited para réplica
    (simulando escritura a S3/Athena/on-premise).
    """
    os.makedirs(output_dir, exist_ok=True)

    cols_replica = ["key_value", "codunicocli", "monto", "tea",
                    "prob_score", "score_tlv", "grupo_ejec_tlv"]
    cols_available = [c for c in cols_replica if c in df.columns]

    filename = f"{table}_{partition}.txt"
    filepath = os.path.join(output_dir, filename)
    df[cols_available].to_csv(filepath, sep="|", index=False)
    print(f"[postprocessing] Réplica guardada en {filepath}")


if __name__ == "__main__":
    print("[postprocessing] Ejecutar desde main.py")
