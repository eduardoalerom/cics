import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# ======================================================
# 1) Configuración e idioma
# ======================================================
st.set_page_config(
    page_title="📊 CICS · INFONAVIT",
    layout="wide",
    initial_sidebar_state="expanded",
)

SP_MONTHS = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}
SP_WEEKDAYS = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}

# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    """Carga robusta del CSV con normalización de columnas y tipos.
    - Detecta separador automáticamente (engine='python', sep=None)
    - Tolera UTF-8 / Latin-1 en ese orden
    - Normaliza nombres de columnas a MAYÚSCULAS sin espacios
    """
    if file is None:
        return pd.DataFrame()

    # Intento UTF-8 → Latin-1
    try:
        df = pd.read_csv(file, engine="python", sep=None, encoding="utf-8")
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, engine="python", sep=None, encoding="latin-1")

    # Nombres de columnas en mayúsculas y sin espacios
    df.columns = df.columns.str.strip().str.upper()

    # Validación de columnas mínimas
    required = {"LPAR", "DATO", "APLICACION", "FECHA", "HORA", "USO"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes: {', '.join(sorted(missing))}.\n\n"
            "Se requieren: LPAR, DATO, APLICACION, FECHA, HORA, USO."
        )

    # Limpiezas consistentes
    df["LPAR"] = df["LPAR"].astype(str).str.upper().str.strip()
    df["DATO"] = (
        df["DATO"].astype(str).str.upper().str.normalize("NFKD")
        .str.replace("Á", "A", regex=False)
        .str.replace("É", "E", regex=False)
        .str.replace("Í", "I", regex=False)
        .str.replace("Ó", "O", regex=False)
        .str.replace("Ú", "U", regex=False)
        .str.strip()
    )
    df["APLICACION"] = df["APLICACION"].astype(str).str.strip().replace({"MIPS": "CPU"})

    # FECHA (dd/mm/yyyy o dd-mm-yyyy)
    df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")

    # HORA ("01:23:45 p. m." → 13:23:45)
    hora_txt = (
        df["HORA"].astype(str)
        .str.replace(" a. m.", " AM", regex=False)
        .str.replace(" p. m.", " PM", regex=False)
        .str.replace("a. m.", " AM", regex=False)
        .str.replace("p. m.", " PM", regex=False)
        .str.strip()
    )
    df["HORA"] = pd.to_datetime(hora_txt, format="%I:%M:%S %p", errors="coerce").dt.time

    # USO numérico
    df["USO"] = pd.to_numeric(df["USO"], errors="coerce").fillna(0)

    # Mes YYYY-MM
    df["MES"] = df["FECHA"].dt.to_period("M").astype(str)

    # Orden útil
    df = df.sort_values(["LPAR", "FECHA", "HORA"], kind="stable").reset_index(drop=True)
    return df


def format_num(value: float, decimals: int = 0) -> str:
    return f"{value:,.{decimals}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def month_label(ym: str) -> str:
    dt = pd.to_datetime(ym + "-01")
    return f"{SP_MONTHS[dt.month]} {dt.year}"


def build_month_options(df: pd.DataFrame) -> list[str]:
    months = sorted(df["MES"].dropna().unique())
    return [month_label(m) for m in months]


def add_trendline_for_cpu(fig: go.Figure, df_m: pd.DataFrame) -> None:
    df_cpu = df_m[df_m["APLICACION"] == "CPU"].copy()
    if len(df_cpu) < 2:
        return
    # De "HH:MM" a número decimal de hora
    hh = df_cpu["HORA"].str.slice(0, 2).astype(int)
    mm = df_cpu["HORA"].str.slice(3, 5).astype(int)
    df_cpu["HORA_NUM"] = hh + mm / 60
    coeffs = np.polyfit(df_cpu["HORA_NUM"], df_cpu["USO"], 1)
    trend = np.polyval(coeffs, df_cpu["HORA_NUM"])  # recta
    fig.add_trace(
        go.Scatter(
            x=df_cpu["HORA"], y=trend, mode="lines", name="Tendencia CPU",
            line=dict(dash="dash", width=2)
        )
    )


# ======================================================
# 2) Encabezado
# ======================================================
st.sidebar.image(
    "https://www.axway.com/sites/default/files/2020-04/Versaria.jpg",
    width=100,
)
st.sidebar.title("📊 Estadísticas CICS")

st.title("INFRAESTRUCTURA Z/OS | INFONAVIT")
st.caption(
    "Este dashboard transforma grandes volúmenes de datos de CICS en métricas y gráficas diarias y mensuales."
)

# ======================================================
# 3) Carga de archivo
# ======================================================
with st.sidebar.expander("📂 Datos", expanded=True):
    uploaded = st.file_uploader(
        "Sube tu archivo (CSV o TXT)", type=["csv", "txt"], help="El separador se detecta automáticamente"
    )

if not uploaded:
    st.info("Sube un archivo para comenzar.")
    st.stop()

# Carga + saneo
try:
    df = load_csv(uploaded)
except Exception as e:
    st.error(f"No se pudo leer el archivo. Detalle: {e}")
    st.stop()

# ======================================================
# 4) Filtros
# ======================================================
months_raw = sorted(df["MES"].unique())
month_labels = [month_label(m) for m in months_raw]

fecha_min = df["FECHA"].min().date()
fecha_max = df["FECHA"].max().date()

with st.sidebar.expander("🧭 Filtros", expanded=True):
    sel_label = st.selectbox("Mes", month_labels, index=len(month_labels) - 1)
    sel_month = months_raw[month_labels.index(sel_label)]

    sel_day = st.date_input(
        "Día", value=fecha_max, min_value=fecha_min, max_value=fecha_max,
        help="Para el detalle horario"
    )

    st.markdown("**LPAR**")
    c1, c2, c3 = st.columns(3)
    opt_sysw = c1.checkbox("SYSW", value=True)
    opt_sysk = c2.checkbox("SYSK", value=True)
    opt_other = c3.checkbox("Otros", value=False)

    selected_envs = []
    if opt_sysw: selected_envs.append("SYSW")
    if opt_sysk: selected_envs.append("SYSK")
    if opt_other:
        selected_envs.extend(sorted([x for x in df["LPAR"].unique() if x not in {"SYSW", "SYSK"}]))
    if not selected_envs:
        st.warning("Selecciona al menos un LPAR.")
        st.stop()

# Subconjunto por mes y LPAR
df_m_lpar = df[(df["MES"] == sel_month) & (df["LPAR"].isin(selected_envs))].copy()
if df_m_lpar.empty:
    st.warning("No hay datos para el mes/LPAR seleccionado.")
    st.stop()

categorias = sorted(df_m_lpar["DATO"].dropna().unique())

# ======================================================
# 5) Acciones (descargas)
# ======================================================
with st.sidebar.expander("⬇️ Exportar"):
    if st.button("📄 Generar PDF del resumen", use_container_width=True):
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            # Portada
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.6, "Reporte INFONAVIT", ha="center", va="center", fontsize=22)
            ax.text(0.5, 0.4, sel_label, ha="center", va="center", fontsize=16)
            ax.axis("off")
            pdf.savefig(fig); plt.close(fig)

            # Barras: uso total por categoría en el mes
            resumen = df_m_lpar.groupby("DATO")["USO"].sum().sort_values()
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            resumen.plot.barh(ax=ax2)
            ax2.set_title("Uso total por categoría (mes)")
            ax2.set_xlabel("USO")
            plt.tight_layout()
            pdf.savefig(fig2); plt.close(fig2)

        buf.seek(0)
        st.sidebar.download_button(
            label="Descargar PDF",
            data=buf,
            file_name=f"reporte_{sel_month}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.download_button(
        "💾 CSV filtrado (mes/LPAR)",
        data=df_m_lpar.to_csv(index=False).encode("utf-8"),
        file_name=f"datos_{sel_month}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ======================================================
# 6) Vista por LPAR con pestañas
# ======================================================
st.markdown("---")
st.subheader(f"Periodo: {sel_label}")

if len(selected_envs) == 1:
    env_tabs = [selected_envs[0]]
else:
    env_tabs = selected_envs

tabs = st.tabs([f"🌐 {e}" for e in env_tabs])
for tab, entorno in zip(tabs, env_tabs):
    with tab:
        df_env = df_m_lpar[df_m_lpar["LPAR"] == entorno]

        # ==================== Métricas del mes ====================
        st.markdown("### 📊 Métricas del mes")
        totales = df_env.groupby("DATO")["USO"].sum()
        cols = st.columns(max(1, min(4, len(categorias))))
        for i, cat in enumerate(categorias):
            val = float(totales.get(cat, 0))
            cols[i % len(cols)].metric(f"{cat}", format_num(val))

        # ==================== Detalle por día (horas) ====================
        st.markdown("---")
        st.markdown(f"### ⏱️ Detalle horario — {sel_day.strftime('%d')} de {sel_label}")
        df_day = df_env[df_env["FECHA"].dt.date == sel_day]

        for cat in categorias:
            st.markdown(f"#### Categoría: {cat}")
            df_cat = df_day[df_day["DATO"] == cat]
            if df_cat.empty:
                st.info("No hay datos para este día.")
                continue

            # Agregados por hora
            agg_h = df_cat.groupby("HORA")["USO"].sum().dropna()
            consumo = float(agg_h.sum())
            if agg_h.empty:
                st.info("Sin horas válidas para graficar.")
                continue

            hr_max = agg_h.idxmax().strftime("%H:%M"); max_v = float(agg_h.max())
            hr_min = agg_h.idxmin().strftime("%H:%M"); min_v = float(agg_h.min())
            top_app = (
                df_cat.groupby("APLICACION")["USO"].sum().sort_values(ascending=False).index[0]
                if not df_cat.empty else "—"
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total del día", format_num(consumo, 2))
            c2.metric("Máx x hora", format_num(max_v, 2), hr_max)
            c3.metric("Mín x hora", format_num(min_v, 2), hr_min)
            c4.metric("App top", top_app)

            multi = {"TRANSACCION", "WEBSERVICE", "MIPS"}
            if cat.upper() in multi:
                df_mh = df_cat.groupby(["HORA", "APLICACION"])["USO"].sum().reset_index()
                df_mh["HORA"] = df_mh["HORA"].apply(lambda t: t.strftime("%H:%M"))
                fig = px.line(
                    df_mh, x="HORA", y="USO", color="APLICACION",
                    title=f"Uso horario de {cat}", markers=True,
                )
                add_trendline_for_cpu(fig, df_mh)
            else:
                df_h = agg_h.reset_index()
                df_h["HORA"] = df_h["HORA"].apply(lambda t: t.strftime("%H:%M"))
                fig = px.line(df_h, x="HORA", y="USO", title=f"Uso horario de {cat}", markers=True)

            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="Aplicación")
            st.plotly_chart(fig, use_container_width=True)

        # ==================== Acumulado diario en el mes ====================
        st.markdown("---")
        st.markdown("### 📈 Acumulado diario en el mes")
        for cat in categorias:
            st.markdown(f"#### Categoría: {cat}")
            df_cm = df_env[df_env["DATO"] == cat].copy()
            if df_cm.empty:
                st.info("No hay datos para esta categoría.")
                continue

            df_cm["DIA"] = df_cm["FECHA"].dt.date
            agg_d = df_cm.groupby("DIA")["USO"].sum().reset_index()
            if agg_d.empty:
                st.info("Sin días válidos para graficar.")
                continue

            agg_d["DIA_SEM"] = agg_d["DIA"].apply(lambda d: SP_WEEKDAYS[pd.Timestamp(d).weekday()])
            fig2 = px.line(agg_d, x="DIA", y="USO", title=f"Acumulado diario: {cat}", markers=True)
            fig2.update_xaxes(tickmode="array", tickvals=agg_d["DIA"], ticktext=agg_d["DIA_SEM"])

            # Anotar pico
            idxmax = agg_d["USO"].idxmax()
            if pd.notna(idxmax):
                pico = agg_d.loc[idxmax]
                fig2.add_trace(
                    go.Scatter(
                        x=[pico["DIA"]], y=[pico["USO"]], mode="markers+text",
                        marker=dict(size=10),
                        text=[f"Máximo {pico['DIA_SEM']} ({format_num(pico['USO'])})"],
                        textposition="top center", showlegend=False,
                    )
                )
            fig2.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig2, use_container_width=True)

# ==================== Footer ====================
st.caption("Hecho con ❤️ por TEAM CICS | Versaria")
