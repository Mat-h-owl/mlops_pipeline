"""
dashboard.py — Dashboard interactivo con Streamlit (+5 pts extra).

Muestra:
  1. Distribución de grupos TLV (grupo_ejec_tlv)
  2. Evolución de AUC y PSI por mes
  3. Tabla top-N clientes por score TLV
  4. Efectividad % y monto promedio por grupo y mes

Ejecutar:
    streamlit run dashboard.py
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Pipeline ML E2E — cu_venta",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data():
    """Carga los archivos generados por el pipeline."""
    data = {}

    # TLV output
    tlv_path = "data/postprocessed/output_tlv.csv"
    if os.path.exists(tlv_path):
        data["tlv"] = pd.read_csv(tlv_path)
    else:
        data["tlv"] = None

    # Monitoring report
    mon_path = "data/monitoring/monitoring_report.json"
    if os.path.exists(mon_path):
        with open(mon_path) as f:
            data["monitoring"] = json.load(f)
    else:
        data["monitoring"] = None

    # Recall by decile
    recall_path = "data/monitoring/recall_by_decile.csv"
    if os.path.exists(recall_path):
        data["recall"] = pd.read_csv(recall_path)
    else:
        data["recall"] = None

    # Best params
    params_path = "models/best_params.json"
    if os.path.exists(params_path):
        with open(params_path) as f:
            data["params"] = json.load(f)
    else:
        data["params"] = None

    # Train/Val/Test sets
    for split in ["df_train", "df_test", "df_val"]:
        path = f"data/processed/{split}.csv"
        if os.path.exists(path):
            data[split] = pd.read_csv(path)
        else:
            data[split] = None

    return data


def main():
    st.title("📊 Pipeline ML E2E — Predicción cu_venta")
    st.markdown("Dashboard de resultados del pipeline XGBoost + Optuna + MLflow")

    data = load_data()

    if data["tlv"] is None:
        st.error(
            "⚠️ No se encontraron datos del pipeline. "
            "Ejecute primero: `python main.py --mode train --input data/raw/dataset.csv`"
        )
        return

    df_tlv = data["tlv"]

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔧 Configuración")

        top_n = st.slider("Top-N clientes", min_value=5, max_value=100,
                          value=20, step=5)

        if data["monitoring"]:
            st.header("📋 Resumen Monitoreo")
            mon = data["monitoring"]
            psi = mon.get("psi_score", 0)
            flag = mon.get("psi_flag", "N/A")
            auc = mon.get("auc_val", 0)

            col1, col2 = st.columns(2)
            col1.metric("PSI", f"{psi:.4f}")
            col2.metric("Estado", flag)
            st.metric("AUC (Validación)", f"{auc:.4f}")

            if flag == "ALERT":
                st.error("🔴 Deriva severa detectada — Re-entrenamiento recomendado")
            elif flag == "WARN":
                st.warning("🟡 Deriva moderada detectada")
            else:
                st.success("🟢 Sin deriva significativa")

        if data["params"]:
            st.header("⚙️ Hiperparámetros")
            for k, v in data["params"].items():
                if k not in ["use_label_encoder", "eval_metric", "random_state"]:
                    st.text(f"{k}: {v}")

    # ── KPIs principales ──────────────────────────────────────────────────
    st.header("📈 Indicadores Principales")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Clientes", f"{len(df_tlv):,}")
    k2.metric("Tasa Conversión", f"{df_tlv['target'].mean():.2%}")
    k3.metric("Score TLV Promedio", f"{df_tlv['score_tlv'].mean():.4f}")
    k4.metric("Monto Promedio", f"S/ {df_tlv['monto'].mean():,.0f}")

    st.divider()

    # ── 1. Distribución de grupos TLV ──────────────────────────────────────
    st.header("1️⃣ Distribución de Grupos de Ejecución TLV")

    col1, col2 = st.columns(2)

    with col1:
        grupo_counts = (
            df_tlv.groupby("grupo_ejec_tlv")
            .size()
            .reset_index(name="n_clientes")
            .sort_values("grupo_ejec_tlv")
        )
        fig1 = px.bar(
            grupo_counts, x="grupo_ejec_tlv", y="n_clientes",
            title="Clientes por Grupo TLV",
            labels={"grupo_ejec_tlv": "Grupo TLV", "n_clientes": "N° Clientes"},
            color="grupo_ejec_tlv",
            color_continuous_scale="RdYlGn_r",
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        grupo_stats = (
            df_tlv.groupby("grupo_ejec_tlv")
            .agg(
                tasa_conversion=("target", "mean"),
                score_tlv_mean=("score_tlv", "mean"),
            )
            .reset_index()
            .sort_values("grupo_ejec_tlv")
        )
        fig2 = px.bar(
            grupo_stats, x="grupo_ejec_tlv", y="tasa_conversion",
            title="Tasa de Conversión por Grupo TLV",
            labels={"grupo_ejec_tlv": "Grupo TLV",
                    "tasa_conversion": "Tasa Conversión"},
            color="tasa_conversion",
            color_continuous_scale="Greens",
        )
        fig2.update_layout(yaxis_tickformat=".1%", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── 2. Métricas de monitoreo ──────────────────────────────────────────
    st.header("2️⃣ Monitoreo: AUC y PSI")

    if data["recall"] is not None:
        col1, col2 = st.columns(2)

        with col1:
            df_recall = data["recall"]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=df_recall["decil"], y=df_recall["recall_acum"],
                mode="lines+markers", name="Recall Acumulado",
                line=dict(color="#2E86C1", width=3),
                marker=dict(size=8),
            ))
            fig3.add_trace(go.Scatter(
                x=df_recall["decil"], y=df_recall["pct_poblacion"],
                mode="lines", name="Línea Base (aleatorio)",
                line=dict(color="gray", dash="dash"),
            ))
            fig3.update_layout(
                title="Recall Acumulado por Decil",
                xaxis_title="Decil (1=mejores)",
                yaxis_title="Recall Acumulado",
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            fig4 = px.bar(
                df_recall, x="decil", y="tasa_conversion",
                title="Tasa de Conversión por Decil",
                labels={"decil": "Decil", "tasa_conversion": "Conversión"},
                color="tasa_conversion",
                color_continuous_scale="YlOrRd",
            )
            fig4.update_layout(yaxis_tickformat=".1%", showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

    if data["monitoring"]:
        mon = data["monitoring"]
        col1, col2, col3 = st.columns(3)
        col1.metric("PSI Score", f"{mon['psi_score']:.4f}",
                    delta=mon['psi_flag'],
                    delta_color="off" if mon['psi_flag'] == "OK" else "inverse")
        col2.metric("AUC Validación", f"{mon['auc_val']:.4f}")
        col3.metric("Re-entrenamiento", "Sí" if mon.get("needs_retrain") else "No")

    st.divider()

    # ── 3. Top-N clientes por score TLV ───────────────────────────────────
    st.header(f"3️⃣ Top-{top_n} Clientes por Score TLV")

    cols_display = ["key_value", "codunicocli", "monto", "tea", "prob_score",
                    "score_tlv", "grupo_ejec_tlv", "target"]
    cols_available = [c for c in cols_display if c in df_tlv.columns]

    top_clientes = (
        df_tlv[cols_available]
        .sort_values("score_tlv", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top_clientes.index = top_clientes.index + 1

    st.dataframe(
        top_clientes.style.format({
            "monto": "S/ {:,.0f}",
            "tea": "{:.2f}%",
            "prob_score": "{:.4f}",
            "score_tlv": "{:.4f}",
        }).background_gradient(subset=["score_tlv"], cmap="YlGn"),
        use_container_width=True,
    )

    st.divider()

    # ── 4. Efectividad y monto por grupo ──────────────────────────────────
    st.header("4️⃣ Efectividad y Monto Promedio por Grupo")

    if "partition" in df_tlv.columns:
        grupo_mes = (
            df_tlv.groupby(["partition", "grupo_ejec_tlv"])
            .agg(
                efectividad=("target", "mean"),
                monto_promedio=("monto", "mean"),
                n_clientes=("target", "size"),
            )
            .reset_index()
        )

        col1, col2 = st.columns(2)

        with col1:
            fig5 = px.line(
                grupo_mes, x="grupo_ejec_tlv", y="efectividad",
                color="partition",
                title="Efectividad (%) por Grupo y Mes",
                labels={"grupo_ejec_tlv": "Grupo TLV",
                        "efectividad": "Efectividad",
                        "partition": "Partición"},
                markers=True,
            )
            fig5.update_layout(yaxis_tickformat=".1%")
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            fig6 = px.bar(
                grupo_mes, x="grupo_ejec_tlv", y="monto_promedio",
                color="partition",
                title="Monto Promedio por Grupo y Mes",
                labels={"grupo_ejec_tlv": "Grupo TLV",
                        "monto_promedio": "Monto Promedio (S/)",
                        "partition": "Partición"},
                barmode="group",
            )
            st.plotly_chart(fig6, use_container_width=True)
    else:
        grupo_stats_full = (
            df_tlv.groupby("grupo_ejec_tlv")
            .agg(
                efectividad=("target", "mean"),
                monto_promedio=("monto", "mean"),
                n_clientes=("target", "size"),
            )
            .reset_index()
        )
        col1, col2 = st.columns(2)
        with col1:
            fig5 = px.bar(
                grupo_stats_full, x="grupo_ejec_tlv", y="efectividad",
                title="Efectividad por Grupo TLV",
                color="efectividad", color_continuous_scale="Greens",
            )
            fig5.update_layout(yaxis_tickformat=".1%", showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)
        with col2:
            fig6 = px.bar(
                grupo_stats_full, x="grupo_ejec_tlv", y="monto_promedio",
                title="Monto Promedio por Grupo TLV",
                color="monto_promedio", color_continuous_scale="Blues",
            )
            fig6.update_layout(showlegend=False)
            st.plotly_chart(fig6, use_container_width=True)

    st.divider()

    # ── 5. Distribución de scores ─────────────────────────────────────────
    st.header("5️⃣ Distribución de Scores")

    if "prob_score" in df_tlv.columns:
        col1, col2 = st.columns(2)

        with col1:
            fig7 = px.histogram(
                df_tlv, x="prob_score", nbins=50,
                color="target", barmode="overlay",
                title="Distribución del Score del Modelo",
                labels={"prob_score": "Probabilidad", "target": "Target"},
                opacity=0.7,
            )
            st.plotly_chart(fig7, use_container_width=True)

        with col2:
            fig8 = px.histogram(
                df_tlv, x="score_tlv", nbins=50,
                title="Distribución del Score TLV",
                labels={"score_tlv": "Score TLV"},
                color_discrete_sequence=["#2E86C1"],
            )
            st.plotly_chart(fig8, use_container_width=True)

    # ── Footer ─────────────────────────────────────────────────────────────
    st.divider()
    st.caption("Pipeline ML E2E — XGBoost + Optuna + MLflow | Maestría en Data Science")


if __name__ == "__main__":
    main()
