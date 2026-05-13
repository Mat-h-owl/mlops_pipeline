# 📊 Pipeline ML E2E — Predicción `cu_venta`

Pipeline end-to-end de Machine Learning para predecir si un cliente bancario comprará un producto financiero (`target = 1` = compró, `target = 0` = no compró), usando **XGBoost** con búsqueda de hiperparámetros via **Optuna** y registro de experimentos en **MLflow**.

## 🎯 Objetivo

El banco tiene una lista de clientes elegibles cada mes, les ofrece un `monto` a una `tea` determinada, y el modelo decide **a quién vale la pena contactar primero**, priorizando mediante un score compuesto TLV (Time-Life Value) y segmentando en 10 grupos de ejecución.

## 📁 Estructura del Proyecto

```
ml-pipeline/
├── main.py               ← Orquestador principal (train / inference)
├── preprocessing.py       ← Limpieza, imputación, encoding, split temporal
├── training.py            ← Entrenamiento XGBoost + Optuna (30 trials) + MLflow
├── monitoring.py          ← PSI, AUC y Recall por decil
├── postprocessing.py      ← Score TLV compuesto + segmentación get_groups()
├── dashboard.py           ← Dashboard interactivo (Streamlit) — Extra +5pts
├── download_data.py       ← Descarga y consolida data desde Google Drive
├── requirements.txt       ← Dependencias del proyecto
├── .gitignore             ← Archivos excluidos de Git
├── README.md              ← Este archivo
├── data/
│   ├── raw/               ← dataset.csv descargado desde Drive
│   ├── processed/         ← df_train.csv, df_test.csv, df_val.csv, df_oot.csv
│   ├── postprocessed/     ← output_tlv.csv
│   ├── replica/           ← Archivos pipe-delimitados (simula S3/Athena)
│   └── monitoring/        ← Reportes JSON y CSV de monitoreo
└── models/
    ├── best_model.joblib  ← Modelo serializado
    ├── best_params.json   ← Mejores hiperparámetros
    └── feature_names.json ← Nombres de features del modelo
```

## 📦 Dataset

- **Fuente:** [Google Drive](https://drive.google.com/drive/u/2/folders/1BbaYLS_Cy5pbvfE6JH3P7KlLdlRf_Cds)
- **Particiones:** p1 a p10 (10 meses de datos)
- **Columnas clave:**
  - `partition` — Período mensual (define el split temporal)
  - `key_value` — Hash único del cliente (se excluye del modelo)
  - `target` — Variable a predecir (0/1)
  - `monto` — Monto del producto ofertado
  - `tea` — Tasa efectiva anual
  - ~60 features numéricas y categóricas

### Split Temporal

| Split | Particiones | Descripción |
|-------|-------------|-------------|
| **Train** | p1 – p7 | Entrenamiento del modelo |
| **Test** | p6 – p7 | Evaluación durante HPO (solapado con train) |
| **Val** | p8 – p9 | Holdout interno para monitoreo |
| **OOT** | p10 | Out-of-Time — simula datos del profesor |

---

## 🚀 Guía de Instalación y Ejecución Paso a Paso

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/<TU_USUARIO>/ml-pipeline.git
cd ml-pipeline
```

### Paso 2: Crear entorno virtual (recomendado)

```bash
python -m venv venv

# En Linux/Mac:
source venv/bin/activate

# En Windows:
venv\Scripts\activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Descargar el dataset

**Opción A — Automática (con `gdown`):**

```bash
python download_data.py
```

Este script descarga todos los archivos CSV de la carpeta de Drive y los consolida en `data/raw/dataset.csv`.

**Opción B — Manual:**

1. Ir a: https://drive.google.com/drive/u/2/folders/1BbaYLS_Cy5pbvfE6JH3P7KlLdlRf_Cds
2. Descargar todos los archivos CSV
3. Colocarlos en `data/raw/`
4. Ejecutar solo la consolidación:
   ```bash
   python download_data.py
   ```

### Paso 5: Ejecutar el pipeline de entrenamiento

```bash
python main.py --mode train --input data/raw/dataset.csv
```

Esto ejecuta las 4 etapas en orden:
1. **Preprocesamiento** → genera splits en `data/processed/`
2. **Entrenamiento** → Optuna (30 trials) + MLflow → modelo en `models/`
3. **Monitoreo** → PSI + AUC + Recall → reportes en `data/monitoring/`
4. **Postprocesamiento** → Score TLV + 10 grupos → `data/postprocessed/`

> **Nota:** La búsqueda con Optuna (30 trials) puede tomar varios minutos.
> Para pruebas rápidas usar `--n_trials 5`.

### Paso 6: Simular inferencia con datos nuevos (OOT)

Para simular lo que hará el profesor con meses que el modelo nunca vio:

```bash
# Separar datos OOT (p10) de los de entrenamiento (p1-p9)
python -c "
import pandas as pd
df = pd.read_csv('data/raw/dataset.csv')
particiones = sorted(df['partition'].unique())
df_oot = df[df['partition'] == particiones[-1]]
df_resto = df[df['partition'].isin(particiones[:-1])]
df_oot.to_csv('data/raw/dataset_oot.csv', index=False)
df_resto.to_csv('data/raw/dataset_train_base.csv', index=False)
print(f'Train base: {len(df_resto):,} filas')
print(f'OOT:        {len(df_oot):,} filas')
"

# Entrenar solo con p1-p9
python main.py --mode train --input data/raw/dataset_train_base.csv

# Inferir con p10 (simula al profesor)
python main.py --mode inference --input data/raw/dataset_oot.csv
```

Si el PSI > 0.25 o los grupos TLV colapsan, el pipeline se **re-entrena automáticamente**.

### Paso 7: Visualizar el dashboard (Extra +5 pts)

```bash
streamlit run dashboard.py
```

Abre el navegador en `http://localhost:8501` y muestra:
1. Distribución de grupos TLV
2. Métricas de monitoreo (AUC, PSI, Recall por decil)
3. Top-N clientes por score TLV
4. Efectividad % y monto promedio por grupo y mes
5. Distribución de scores (modelo y TLV)

### Paso 8: Ver experimentos en MLflow (opcional)

```bash
mlflow ui
```

Abre `http://localhost:5000` para ver los experimentos registrados, comparar parámetros y métricas.

---

## 🔄 Lógica de Re-entrenamiento Automático

El pipeline detecta automáticamente cuándo el modelo necesita re-entrenarse:

| Condición | Umbral | Acción |
|-----------|--------|--------|
| PSI > 0.25 | Deriva severa | Re-entrena automáticamente |
| Grupos TLV colapsan | < 5 grupos distintos | Re-entrena automáticamente |
| PSI 0.10 – 0.25 | Deriva moderada | Warning (no re-entrena) |
| PSI < 0.10 | Sin deriva | OK |

---

## 📐 Las 5 Etapas del Pipeline

### 1. `preprocessing.py`
- Carga el CSV crudo y detecta automáticamente columnas categóricas/numéricas
- Imputa nulos (mediana para numéricas, moda para categóricas)
- Label-encoding de categóricas (fit en train, transform en todo)
- Split temporal por `partition` (NO aleatorio)
- Guarda metadata (encoders, nombres de columnas) para modo inferencia

### 2. `training.py`
- Busca hiperparámetros con **Optuna** (30 trials): n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda
- Entrena **XGBoost** final con los mejores parámetros
- Registra en **MLflow**: parámetros, AUC test/val, modelo serializado
- Registra el modelo en MLflow Model Registry como `cu_venta_xgb`
- Guarda modelo `.joblib` local para inferencia rápida

### 3. `monitoring.py`
- Calcula **PSI** (Population Stability Index) entre distribuciones de score train vs val
- Evalúa AUC sobre validación
- Genera tabla de **Recall acumulado por decil** (decil 1 = mejores prospectos)
- Guarda reporte JSON y CSV en `data/monitoring/`

### 4. `postprocessing.py`
- Calcula score compuesto **TLV** = 0.50 × prob_score + 0.30 × monto_norm + 0.20 × tea_norm
- Segmenta en 10 grupos (`grupo_ejec_tlv`) usando **`get_groups()`** del curso **tal cual**, sin modificar la fórmula ni los cuantiles DIST_GE
- Valida que los grupos no hayan colapsado
- Genera réplica en formato pipe-delimited (`data/replica/`)

### 5. `main.py`
- Orquesta las 4 etapas en orden
- Soporta modo `train` e `inference`
- Implementa lógica de **re-entrenamiento automático** si PSI > 0.25 o grupos colapsan

---

## 📊 Dashboard (Extra +5 pts)

Implementado en **Streamlit** con **Plotly** para gráficos interactivos.

```bash
# Primero correr el pipeline (genera los archivos)
python main.py --mode train --input data/raw/dataset.csv

# Luego visualizar los resultados
streamlit run dashboard.py
```

### Publicación en Streamlit Cloud

1. Subir el repo a GitHub
2. Ir a [share.streamlit.io](https://share.streamlit.io)
3. Conectar el repositorio
4. Seleccionar `dashboard.py` como archivo principal
5. Deploy → obtener URL pública

**URL del dashboard:** `[PENDIENTE — agregar después del deploy]`

---

## 🛠️ Requisitos del Sistema

- Python 3.9+
- pip (gestor de paquetes)
- ~4 GB de RAM (para Optuna + XGBoost)
- Conexión a internet (para descargar datos de Drive)

---

## 📝 Notas

- Los datos NO se suben a GitHub (ver `.gitignore`). El evaluador debe descargarlos con `download_data.py` o manualmente desde Drive.
- MLflow almacena los experimentos localmente en `mlruns/`. No se sube a GitHub.
- El modelo entrenado (`.joblib`) tampoco se sube; se regenera ejecutando el pipeline.
- La columna `prob_value_contact` del dataset original NO se usa como feature del modelo (podría ser un score pre-existente que generaría data leakage).

---

## 👤 Autor

- **Nombre:** [Tu nombre completo]
- **Curso:** Maestría en Data Science
- **Repositorio:** https://github.com/<TU_USUARIO>/ml-pipeline
