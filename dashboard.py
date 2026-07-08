"""
dashboard.py
Dashboard web — Cable Visión IA
Ejecutar: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
import sys, os, base64

sys.path.insert(0, os.path.dirname(__file__))
import database as db

# ── Configuración de página ───────────────────────────────────
st.set_page_config(
    page_title="Cable Visión IA",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilo ────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card { background:#1e1e2e; border-radius:10px; padding:16px; text-align:center; }
.metric-val  { font-size:2rem; font-weight:700; color:#7c83fd; }
.metric-lbl  { font-size:.85rem; color:#aaa; }
</style>
""", unsafe_allow_html=True)

# ── Carga de datos ────────────────────────────────────────────
@st.cache_data(ttl=300)
def cargar_datos():
    perfil    = db.query_df("SELECT * FROM vw_PerfilClienteChurn")
    stock     = db.query_df("SELECT * FROM vw_AlertaStockCritico WHERE EstadoStock IN ('CRITICO','BAJO','EXCESO')")
    try:
        segmentos = db.query_df("""
            SELECT TOP 1000 s.ClienteID, s.NombreSegmento, s.ClusterID
            FROM SegmentosCliente s
            INNER JOIN (
                SELECT ClienteID, MAX(FechaAsignacion) AS ult
                FROM SegmentosCliente GROUP BY ClienteID
            ) u ON s.ClienteID = u.ClienteID AND s.FechaAsignacion = u.ult
        """)
    except Exception:
        segmentos = pd.DataFrame(columns=["ClienteID","NombreSegmento","ClusterID"])
    try:
        predicciones = db.query_df("""
            SELECT TOP 1000 p.ClienteID, p.ScoreChurn, p.NivelRiesgo, p.FechaPrediccion
            FROM PrediccionesChurn p
            INNER JOIN (
                SELECT ClienteID, MAX(FechaPrediccion) AS ult
                FROM PrediccionesChurn GROUP BY ClienteID
            ) u ON p.ClienteID = u.ClienteID AND p.FechaPrediccion = u.ult
        """)
    except Exception:
        # Leer del CSV si no hay predicciones en BD
        exports = os.path.join(os.path.dirname(__file__), "exports")
        csvs = sorted([f for f in os.listdir(exports) if f.startswith("predicciones_churn")])
        if csvs:
            predicciones = pd.read_csv(os.path.join(exports, csvs[-1]))
        else:
            predicciones = pd.DataFrame(columns=["ClienteID","ScoreChurn","NivelRiesgo"])
    return perfil, stock, segmentos, predicciones

try:
    perfil, stock, segmentos, predicciones = cargar_datos()
    conexion_ok = True
except Exception as e:
    st.error(f"Error al conectar con la base de datos: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), "img", "icono.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f'<img src="data:image/png;base64,{logo_b64}" style="width:100%;max-width:220px;margin-bottom:8px">',
            unsafe_allow_html=True
        )
    else:
        st.title("📡 Cable Visión IA")

    st.caption("UPSJB — Inteligencia Artificial 2026")
    st.divider()

    pagina = option_menu(
        menu_title=None,
        options=[
            "Resumen General",
            "Predicción de Churn",
            "Segmentación",
            "Stock Crítico",
            "Comparación de Modelos",
        ],
        icons=["house", "graph-down", "people", "box-seam", "bar-chart"],
        menu_icon="list",
        default_index=0,
        styles={
            "container":      {"padding": "0", "background-color": "transparent"},
            "icon":           {"color": "#7c83fd", "font-size": "15px"},
            "nav-link":       {"font-size": "14px", "text-align": "left", "margin": "2px 0",
                               "--hover-color": "#2a2a3e"},
            "nav-link-selected": {"background-color": "#7c83fd", "color": "white"},
        },
    )

    st.divider()
    if st.button("🔄 Actualizar datos"):
        st.cache_data.clear()
        st.rerun()

# ════════════════════════════════════════════════════════════════
# PÁGINA 1 — RESUMEN GENERAL
# ════════════════════════════════════════════════════════════════
if pagina == "Resumen General":
    st.title("📡 Cable Visión IA — Resumen General")

    total     = len(perfil)
    activos   = len(perfil[perfil["EstadoCliente"] == "Activo"]) if "EstadoCliente" in perfil.columns else 0
    churn_n   = int(perfil["Churn"].sum()) if "Churn" in perfil.columns else 0
    tasa_churn = round(churn_n / total * 100, 1) if total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total clientes", f"{total:,}")
    c2.metric("Clientes activos", f"{activos:,}")
    c3.metric("Churn confirmado", f"{churn_n:,}")
    c4.metric("Tasa de churn", f"{tasa_churn}%")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Estado de clientes")
        if "EstadoCliente" in perfil.columns:
            estado_cnt = perfil["EstadoCliente"].value_counts().reset_index()
            estado_cnt.columns = ["Estado", "Cantidad"]
            fig = px.pie(estado_cnt, values="Cantidad", names="Estado",
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribución de churn")
        if "Churn" in perfil.columns:
            churn_cnt = perfil["Churn"].map({0: "No Churn", 1: "Churn"}).value_counts().reset_index()
            churn_cnt.columns = ["Churn", "Cantidad"]
            fig = px.bar(churn_cnt, x="Churn", y="Cantidad",
                         color="Churn",
                         color_discrete_map={"Churn": "#e74c3c", "No Churn": "#2ecc71"},
                         text="Cantidad")
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Distribución de features clave
    st.subheader("Distribución de features principales")
    col3, col4 = st.columns(2)
    features_disp = [c for c in ["MesesComoCliente","PrecioMensual","TotalIncidencias","PromedioAtraso"] if c in perfil.columns]
    for i, feat in enumerate(features_disp):
        col = col3 if i % 2 == 0 else col4
        with col:
            fig = px.histogram(perfil, x=feat, nbins=20,
                               color_discrete_sequence=["#7c83fd"],
                               title=feat)
            fig.update_layout(margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PÁGINA 2 — PREDICCIÓN DE CHURN
# ════════════════════════════════════════════════════════════════
elif pagina == "Predicción de Churn":
    st.title("📉 Predicción de Churn")

    if predicciones.empty:
        st.warning("No hay predicciones disponibles. Ejecuta primero `python main.py`.")
    else:
        st.write("**Filtrar por nivel de riesgo:**")
        col_a, col_b, col_c = st.columns(3)
        f_alto  = col_a.checkbox("🔴 Alto",  value=True)
        f_medio = col_b.checkbox("🟡 Medio", value=True)
        f_bajo  = col_c.checkbox("🟢 Bajo",  value=False)
        nivel_filtro = (["Alto"]  if f_alto  else []) + \
                       (["Medio"] if f_medio else []) + \
                       (["Bajo"]  if f_bajo  else [])
        df_f = predicciones[predicciones["NivelRiesgo"].isin(nivel_filtro)] if nivel_filtro else predicciones

        c1, c2, c3 = st.columns(3)
        c1.metric("Alto riesgo",  len(predicciones[predicciones["NivelRiesgo"]=="Alto"]))
        c2.metric("Medio riesgo", len(predicciones[predicciones["NivelRiesgo"]=="Medio"]))
        c3.metric("Bajo riesgo",  len(predicciones[predicciones["NivelRiesgo"]=="Bajo"]))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Clientes por nivel de riesgo")
            nivel_cnt = predicciones["NivelRiesgo"].value_counts().reset_index()
            nivel_cnt.columns = ["Nivel", "Cantidad"]
            fig = px.pie(nivel_cnt, values="Cantidad", names="Nivel",
                         color="Nivel",
                         color_discrete_map={"Alto":"#e74c3c","Medio":"#f39c12","Bajo":"#2ecc71"},
                         hole=0.35)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Distribución de Score Churn")
            fig = px.histogram(predicciones, x="ScoreChurn", nbins=30,
                               color_discrete_sequence=["#e74c3c"],
                               title="Histograma de scores")
            fig.add_vline(x=0.40, line_dash="dash", line_color="orange", annotation_text="Umbral Medio")
            fig.add_vline(x=0.65, line_dash="dash", line_color="red",    annotation_text="Umbral Alto")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Tabla de predicciones ({len(df_f)} clientes)")
        st.dataframe(
            df_f.sort_values("ScoreChurn", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=400
        )


# ════════════════════════════════════════════════════════════════
# PÁGINA 3 — SEGMENTACIÓN
# ════════════════════════════════════════════════════════════════
elif pagina == "Segmentación":
    st.title("👥 Segmentación de Clientes (K-Means)")

    exports = os.path.join(os.path.dirname(__file__), "exports")
    csvs = sorted([f for f in os.listdir(exports) if f.startswith("segmentos_clientes")])
    if not csvs:
        st.warning("No hay segmentos. Ejecuta `python main.py --solo-segmentos`.")
    else:
        seg = pd.read_csv(os.path.join(exports, csvs[-1]))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Clientes por segmento")
            cnt = seg["NombreSegmento"].value_counts().reset_index()
            cnt.columns = ["Segmento", "Cantidad"]
            fig = px.bar(cnt, x="Segmento", y="Cantidad",
                         color="Segmento",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         text="Cantidad")
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Proporción de segmentos")
            fig = px.pie(cnt, values="Cantidad", names="Segmento",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        # Scatter si hay features numéricas
        num_cols = [c for c in seg.columns if seg[c].dtype in ["float64","int64"] and c not in ["ClienteID","ClusterID"]]
        if len(num_cols) >= 2:
            st.subheader("Dispersión de segmentos")
            cx, cy = st.columns(2)
            x_col = cx.selectbox("Eje X", num_cols, index=0)
            y_col = cy.selectbox("Eje Y", num_cols, index=min(1, len(num_cols)-1))
            fig = px.scatter(seg, x=x_col, y=y_col, color="NombreSegmento",
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             hover_data=["ClienteID"])
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Datos de segmentos")
        st.dataframe(seg, use_container_width=True, height=350)


# ════════════════════════════════════════════════════════════════
# PÁGINA 4 — STOCK CRÍTICO
# ════════════════════════════════════════════════════════════════
elif pagina == "Stock Crítico":
    st.title("📦 Inventario — Stock Crítico")

    if stock.empty:
        st.success("✅ No hay equipos con stock crítico o bajo.")
    else:
        critico = len(stock[stock["EstadoStock"] == "CRITICO"])
        bajo    = len(stock[stock["EstadoStock"] == "BAJO"])
        exceso  = len(stock[stock["EstadoStock"] == "EXCESO"])

        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 Crítico", critico)
        c2.metric("🟡 Bajo",    bajo)
        c3.metric("🔵 Exceso",  exceso)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Stock actual vs mínimo por equipo")
            fig = go.Figure()
            fig.add_bar(name="Stock Actual", x=stock["TipoEquipo"] + " — " + stock["Zona"],
                        y=stock["CantidadActual"], marker_color="#7c83fd")
            fig.add_bar(name="Stock Mínimo", x=stock["TipoEquipo"] + " — " + stock["Zona"],
                        y=stock["StockMinimo"], marker_color="#e74c3c")
            fig.update_layout(barmode="group", xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Estado de stock")
            cnt = stock["EstadoStock"].value_counts().reset_index()
            cnt.columns = ["Estado","Cantidad"]
            fig = px.pie(cnt, values="Cantidad", names="Estado",
                         color="Estado",
                         color_discrete_map={"CRITICO":"#e74c3c","BAJO":"#f39c12","EXCESO":"#3498db"},
                         hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detalle de equipos")
        st.dataframe(stock, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PÁGINA 5 — COMPARACIÓN DE MODELOS
# ════════════════════════════════════════════════════════════════
elif pagina == "Comparación de Modelos":
    st.title("📊 Comparación de Modelos IA")

    exports = os.path.join(os.path.dirname(__file__), "exports")
    csvs = sorted([f for f in os.listdir(exports) if f.startswith("comparacion_modelos")])
    if not csvs:
        st.warning("No hay datos de comparación. Ejecuta `python main.py`.")
    else:
        comp = pd.read_csv(os.path.join(exports, csvs[-1]))
        comp = comp.reset_index() if "nombre" not in comp.columns else comp.rename(columns={"nombre":"Modelo"})

        st.subheader("Métricas por modelo")
        metricas = [c for c in comp.columns if c in ["precision","recall","f1","auc_roc"]]

        if metricas:
            fig = go.Figure()
            colores = ["#7c83fd","#2ecc71","#f39c12","#e74c3c"]
            for i, met in enumerate(metricas):
                fig.add_bar(
                    name=met.upper(),
                    x=comp.iloc[:,0],
                    y=comp[met],
                    marker_color=colores[i % len(colores)],
                    text=comp[met].round(4),
                    textposition="outside"
                )
            fig.update_layout(barmode="group", yaxis_range=[0, 1.1],
                              xaxis_tickangle=-15)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tabla completa")
        st.dataframe(comp, use_container_width=True)

        # Radar chart
        if len(metricas) >= 3 and len(comp) > 0:
            st.subheader("Radar de métricas")
            fig = go.Figure()
            for _, row in comp.iterrows():
                vals = [float(row[m]) if pd.notna(row[m]) else 0 for m in metricas]
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=metricas + [metricas[0]],
                    fill="toself",
                    name=str(row.iloc[0])
                ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])))
            st.plotly_chart(fig, use_container_width=True)
