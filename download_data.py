"""
download_data.py — Descarga y consolida el dataset desde Google Drive.

El dataset está dividido en 10 partes (archivos CSV) en la carpeta de Drive.
Este script descarga todos los archivos y los consolida en un solo dataset.csv.

Uso:
    python download_data.py
"""

import os
import glob
import pandas as pd

try:
    import gdown
except ImportError:
    print("Instalando gdown...")
    os.system("pip install gdown")
    import gdown


# ── Configuración ──────────────────────────────────────────────────────────
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1BbaYLS_Cy5pbvfE6JH3P7KlLdlRf_Cds"
RAW_DIR = "data/raw"
OUTPUT_FILE = os.path.join(RAW_DIR, "dataset.csv")


def download_from_drive():
    """Descarga todos los archivos CSV de la carpeta de Google Drive."""
    os.makedirs(RAW_DIR, exist_ok=True)

    print(f"[download] Descargando archivos desde Google Drive...")
    print(f"[download] URL: {DRIVE_FOLDER_URL}")

    # Descargar toda la carpeta
    gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        output=RAW_DIR,
        quiet=False,
        use_cookies=False,
    )

    print(f"[download] Archivos descargados en {RAW_DIR}/")


def consolidate_csvs():
    """
    Consolida todos los CSVs descargados en un solo dataset.csv.
    Busca archivos .csv en data/raw/ y los concatena.
    """
    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))

    # Excluir el output consolidado si ya existe
    csv_files = [f for f in csv_files if "dataset.csv" not in f]

    if not csv_files:
        print("[download] ⚠️ No se encontraron archivos CSV en data/raw/")
        print("[download] Si la descarga automática no funcionó, descargue")
        print(f"           manualmente desde: {DRIVE_FOLDER_URL}")
        print(f"           y coloque los archivos en {RAW_DIR}/")
        return None

    print(f"[download] Archivos encontrados ({len(csv_files)}):")
    for f in csv_files:
        print(f"  → {os.path.basename(f)}")

    # Concatenar todos los CSVs
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"  ✓ {os.path.basename(f)}: {len(df):,} filas")
        except Exception as e:
            print(f"  ✗ {os.path.basename(f)}: Error — {e}")

    if not dfs:
        print("[download] No se pudo leer ningún archivo.")
        return None

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[download] Dataset consolidado: {len(df_all):,} filas × {len(df_all.columns)} cols")
    print(f"[download] Guardado en: {OUTPUT_FILE}")

    # Resumen de particiones
    if "partition" in df_all.columns:
        part_counts = df_all["partition"].value_counts().sort_index()
        print(f"\n[download] Particiones:")
        for p, n in part_counts.items():
            print(f"  {p}: {n:,} filas")

    return df_all


if __name__ == "__main__":
    download_from_drive()
    consolidate_csvs()
