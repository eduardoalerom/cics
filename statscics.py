import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1) Configuración de la página
# -------------------------------------------------------
st.set_page_config(
    page_title="📊 CICS · INFONAVIT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# 2) Título y descripción
# -------------------------------------------------------
st.sidebar.image(
    "https://www.axway.com/sites/default/files/2020-04/Versaria.jpg",
    width=100
)
st.sidebar.title("📊  Estadísticas CICS")

st.title("INFRAESTRUCTURA Z/OS | INFONAVIT")
st.markdown(
    """
    
    Este dashboard convierte grandes volúmenes de datos de CICS en gráficos y métricas
    que muestran tendencias diarias y mensuales.
    
    """
)

# -------------------------------------------------------
# 3) Carga del archivo CSV (rápida y robusta)
#   - Evitamos autodetección lenta (sep=None/engine='python')
#   - Tipos explícitos y parseo directo
#   - Cache de lectura para archivos grandes
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    # Leer con separador fijo y tipos simples
    df = pd.read_csv(
        BytesIO(file_bytes),
        encoding="latin-1",
        sep=",",
        usecols=["LPAR", "DATO", "FECHA", "HORA", "APLICACION", "USO"],
        dtype={
            "LPAR": "string",
            "DATO": "string",
            "FECHA": "string",
            "HORA": "string",
            "APLICACION": "string",
            # USO puede venir muy grande; float64 es seguro para agregaciones
            "USO": "float64",
        },
        low_memory=False,
    )

    # Limpieza mínima (sin normalizaciones costosas)
    df["LPAR"] = df["LPAR"].str.strip().str.upper()
    df["DATO"] = df["DATO"].str.strip().str.upper()
    df["APLICACION"] = df["APLICACION"].str.strip()

    # FECHA: día/mes/año o año-mes-día → detecta y convierte rápido
    df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")

    # HORA: 24h HH:MM:SS → a tipo time (para métricas) y a string HH:MM (para ejes)
    h = pd.to_datetime(df["HORA"], format="%H:%M:%S", errors="coerce")
    df["HORA_TIME"] = h.dt.time
    df["HORA_STR"] = h.dt.strftime("%H:%M")

    # Mapear DATO compacto a etiqueta legible
    dato_map = {"T": "TRANSACCION", "W": "WEBSERVICE", "M": "MIPS"}
    df["DATO"] = df["DATO"].map(dato_map).fillna(df["DATO"])  # en caso de valores atípicos

    # Para MIPS, homogeneizar nombre de aplicación como CPU (si aplica)
    df.loc[df["DATO"].eq("MIPS"), "APLICACION"] = df.loc[df["DATO"].eq("MIPS"), "APLICACION"].fillna("CPU").replace("", "CPU")

    # Mes (AAAA-MM) para filtros
    df["MES"] = df["FECHA"].dt.to_period("M").astype(str)

    # Tipos categóricos (menos memoria y más velocidad en groupby)
    for col in ("LPAR", "DATO", "APLICACION", "MES", "HORA_STR"):
        df[col] = df[col].astype("category")

    return df

uploaded_file = st.file_uploader("📂 Sube tu archivo: ", type=["csv"])  # nuevo formato
st.markdown("---")
if uploaded_file is None:
    st.stop()

df = load_csv_bytes(uploaded_file.getvalue())

# -------------------------------------------------------
# 4) Filtros (dinámicos y ligeros)
# -------------------------------------------------------
# Meses en español
meses_es = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

meses_disponibles = sorted(df["MES"].dropna().astype(str).unique())
labels_meses = []
for m in meses_disponibles:
    dt = pd.to_datetime(f"{m}-01")
    labels_meses.append(f"{meses_es[dt.month]} {dt.year}")

idx_default_mes = len(labels_meses) - 1 if labels_meses else 0
mes_seleccionado_label = st.sidebar.selectbox("Selecciona MES", labels_meses, index=idx_default_mes)
mes_seleccionado = meses_disponibles[labels_meses.index(mes_seleccionado_label)] if labels_meses else None

# Día dentro del rango (del dataset completo)
fecha_min = df["FECHA"].dt.date.min()
fecha_max = df["FECHA"].dt.date.max()
dia_seleccionado = st.sidebar.date_input(
    "Selecciona DÍA",
    value=fecha_max,
    min_value=fecha_min,
    max_value=fecha_max
)

st.sidebar.markdown("---")
# -------------------------------------------------------
# 5) Exportación rápida a PDF (opcional)
# -------------------------------------------------------
if st.sidebar.button("📄 Exportar como PDF"):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # Portada
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Reporte INFONAVIT\n{mes_seleccionado_label}",
                ha='center', va='center', fontsize=24)
        ax.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        # Resumen mensual por DATO (todas las LPAR)
        df_resumen = df[df["MES"] == mes_seleccionado]
        resumen = df_resumen.groupby("DATO", observed=True)["USO"].sum().sort_values()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        resumen.plot.barh(ax=ax2)
        ax2.set_title("Uso Total por Categoría (Mes)")
        ax2.set_xlabel("USO")
        plt.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

    buffer.seek(0)
    st.sidebar.download_button(
        label="Descargar PDF",
        data=buffer,
        file_name=f"reporte_{mes_seleccionado}.pdf",
        mime="application/pdf"
    )

# -------------------------------------------------------
# 6) Datos filtrados por MES
# -------------------------------------------------------
df_mes = df[df["MES"] == mes_seleccionado]

# LPARs disponibles en el mes seleccionado → pestañas en la vista principal
lpars_mes = sorted(df_mes["LPAR"].dropna().unique().tolist())


# Categorías ordenadas y legibles
orden_categorias = ["TRANSACCION", "WEBSERVICE", "MIPS"]

# Mapa de días de semana en español
weekday_map = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}

# -------------------------------------------------------
# 7) Panel por LPAR en pestañas
# -------------------------------------------------------
if not lpars_mes:
    st.info("No hay LPARs con datos en el mes seleccionado.")
else:
    tabs = st.tabs([f"⚙️  LPAR: {l}" for l in lpars_mes])
    for tab, entorno in zip(tabs, lpars_mes):
        with tab:
            st.header(f"🌐 Entorno: {entorno}")
            df_entorno = df_mes[df_mes["LPAR"] == entorno]

            # Categorías presentes en este entorno/mes
            categorias = [c for c in orden_categorias if c in df_entorno["DATO"].unique().tolist()]

            # ===== Estilos para tarjetas de métricas con degradado =====
            st.markdown("""
            <style>
            /* Contenedor en una sola fila con scroll horizontal */
            .metric-row {
              display: flex;
              flex-wrap: nowrap;
              gap: 14px;
              overflow-x: auto;
              -webkit-overflow-scrolling: touch;
              padding-bottom: 8px;
              margin-top: 6px;
            }
            .metric-row::-webkit-scrollbar { height: 10px; }
            .metric-row::-webkit-scrollbar-thumb { background: rgba(0,0,0,.25); border-radius: 12px; }
            .metric-row::-webkit-scrollbar-track { background: rgba(0,0,0,.07); }
            
            /* Tarjeta con degradado y efectos */
            .metric-card {
              flex: 0 0 auto;              /* No se encoge, permanece en una sola fila */
              min-width: 240px;
              max-width: 320px;
              position: relative;
              padding: 16px 18px;
              border-radius: 16px;
              color: #ffffff;
              box-shadow: 0 8px 22px rgba(0,0,0,.18);
              border: 1px solid rgba(255,255,255,.18);
              transition: transform .25s ease, box-shadow .25s ease, filter .25s ease;
              overflow: hidden;
            }
            .metric-card:hover {
              transform: translateY(-4px);
              box-shadow: 0 14px 28px rgba(0,0,0,.28);
              filter: saturate(1.08);
            }
            .metric-title { font-weight: 600; opacity: .95; margin-bottom: 8px; }
            .metric-value { font-size: 34px; font-weight: 800; line-height: 1; letter-spacing: .4px; }
            .metric-sub { font-size: 12px; opacity: .9; margin-top: 4px; }
            
            /* Efecto shimmer sutil */
            .metric-card.shimmer::after {
              content: "";
              position: absolute; top: 0; left: -120%;
              width: 50%; height: 100%;
              background: linear-gradient(120deg, rgba(255,255,255,0), rgba(255,255,255,.30), rgba(255,255,255,0));
              transform: skewX(-20deg);
              animation: shimmer 4s ease-in-out infinite;
            }
            @keyframes shimmer { 0% { left: -120%; } 60% { left: 130%; } 100% { left: 130%; } }
            
            /* Paletas con degradado */
            .grad-blue   { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); }
            .grad-green  { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
            .grad-purple { background: linear-gradient(135deg, #8e2de2 0%, #4a00e0 100%); }
            .grad-orange { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); }
            </style>
            """, unsafe_allow_html=True)
            
            def metric_card_html(title: str, value: str, subtitle: str = "",
                                 gradient_cls: str = "grad-green", shimmer: bool = True) -> str:
                shimmer_cls = "shimmer" if shimmer else ""
                # HTML "compacto" sin indentación para que Markdown no lo trate como código
                return (
                    f"<div class='metric-card {gradient_cls} {shimmer_cls}'>"
                    f"<div class='metric-title'>{title}</div>"
                    f"<div class='metric-value'>{value}</div>"
                    f"<div class='metric-sub'>{subtitle}</div>"
                    f"</div>"
                        )

            
            # ==========================
            #  Sustituye tu fragmento por esto
            #  (usa tus variables existentes: df_entorno, categorias, mes_seleccionado_label)
            # ==========================
            
            # Métricas totales del mes por categoría
            st.subheader(f"📊 Métricas totales del Mes de {mes_seleccionado_label}")
            df_totales = df_entorno.groupby("DATO", observed=True)["USO"].sum()
            
            # Mapeo de categoría -> degradado (ajusta a tus categorías reales)
            grad_map = {
                "TRANSACCION": "grad-blue",
                "WEBSERVICE":  "grad-green",
                "MIPS":        "grad-purple",
                # por defecto usará grad-orange si no está en el mapa
            }
            
            if categorias:
                # Construimos todas las tarjetas y las mostramos en UNA sola fila (scroll si hay muchas)
                cards_html = []
                for cat in categorias:
                    total = float(df_totales.get(cat, 0.0))
                    cards_html.append(
                        metric_card_html(
                            title=f"Uso total de {cat}",
                            value=f"{total:,.0f}",
                            subtitle=mes_seleccionado_label,
                            gradient_cls=grad_map.get(cat, "grad-orange"),
                            shimmer=True  # pon False si no quieres el brillo
                        )
                    )
                st.markdown(f'<div class="metric-row">{"".join(cards_html)}</div>', unsafe_allow_html=True)
            else:
                st.info("No hay categorías en el mes seleccionado.")

            # Detalle Diario
            st.markdown("---")
            st.subheader(f"📊 Detalle por día - {dia_seleccionado.strftime('%d')} de {mes_seleccionado_label}")
            df_dia = df_entorno[df_entorno["FECHA"].dt.date == dia_seleccionado]

            for cat in categorias:
                st.markdown(f"### Categoría: {cat}")
                df_cat = df_dia[df_dia["DATO"] == cat]
                if df_cat.empty:
                    st.info("No hay datos.")
                    st.markdown("---")
                    continue

                consumo = float(df_cat["USO"].sum())
                agg_h = df_cat.groupby("HORA_TIME", observed=True)["USO"].sum()
                # Horas extremos (manejar posibles NaT)
                if not agg_h.empty and agg_h.index.notna().any():
                    agg_h_valid = agg_h[agg_h.index.notna()]
                    max_v = float(agg_h_valid.max()); hr_max = agg_h_valid.idxmax().strftime("%H:%M")
                    min_v = float(agg_h_valid.min()); hr_min = agg_h_valid.idxmin().strftime("%H:%M")
                else:
                    max_v = min_v = 0.0; hr_max = hr_min = "--:--"

                # App top
                top_app = df_cat.groupby("APLICACION", observed=True)["USO"].sum()
                top_app_name = top_app.idxmax() if not top_app.empty else "-"

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total del día", f"{consumo:,.2f}")
                c2.metric("Máx x hora", f"{max_v:,.2f}", hr_max)
                c3.metric("Mín x hora", f"{min_v:,.2f}", hr_min)
                c4.metric("App top", str(top_app_name))

                # Serie por hora y aplicación (ejes como HH:MM ya precalculado)
                df_m = (
                    df_cat.groupby(["HORA_STR", "APLICACION"], observed=True)["USO"].sum().reset_index()
                )
                fig = px.line(
                    df_m, x="HORA_STR", y="USO", color="APLICACION",
                    title=f"Uso horario de {cat}", markers=True
                )

                # Tendencia (solo si existe serie 'CPU')
                df_cpu = df_m[df_m["APLICACION"] == "CPU"].copy()
                if len(df_cpu) > 1:
                    # Convertir HH:MM a número de hora (para regresión simple)
                    hh = df_cpu["HORA_STR"].str.slice(0, 2).astype(int)
                    mm = df_cpu["HORA_STR"].str.slice(3, 5).astype(int)
                    hora_num = hh + mm / 60.0
                    coeffs = np.polyfit(hora_num.to_numpy(), df_cpu["USO"].to_numpy(), 1)
                    trend = np.polyval(coeffs, hora_num.to_numpy())
                    fig.add_trace(
                        go.Scatter(
                            x=df_cpu["HORA_STR"],
                            y=trend,
                            mode="lines",
                            name="Tendencia CPU",
                            line=dict(dash="dash", width=2)
                        )
                    )

                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")

            # Acumulado diario del mes (x día)
            st.subheader(f"📈 Acumulado diario del Mes de {mes_seleccionado_label}")
            for cat in categorias:
                st.markdown(f"### Categoría: {cat}")
                df_cm = df_entorno[df_entorno["DATO"] == cat]
                if df_cm.empty:
                    st.info("No hay datos.")
                    st.markdown("---")
                    continue

                agg_d = df_cm.groupby(df_cm["FECHA"].dt.date, observed=True)["USO"].sum().reset_index(name="USO")
                agg_d.rename(columns={"FECHA": "DIA"}, inplace=True)
                agg_d["DIA_SEM"] = agg_d["DIA"].apply(lambda d: weekday_map[pd.Timestamp(d).weekday()])

                fig2 = px.line(agg_d, x="DIA", y="USO", title=f"Acumulado diario: {cat}", markers=True)
                fig2.update_xaxes(
                    tickmode="array",
                    tickvals=agg_d["DIA"],
                    ticktext=agg_d["DIA_SEM"]
                )

                pico = agg_d.loc[agg_d["USO"].idxmax()]
                fig2.add_trace(
                    go.Scatter(
                        x=[pico["DIA"]], y=[pico["USO"]],
                        mode="markers+text",
                        marker=dict(size=10, color="red"),
                        text=[f"Máximo el día {pico['DIA_SEM']} ({pico['USO']:,.0f})"],
                        textposition="top center",
                        showlegend=False
                    )
                )

                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("---")

# ==================== Footer ====================
st.caption("Hecho con ❤️ por TEAM CICS | Versaria")
