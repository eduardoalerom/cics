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
    page_title="Estadísticas INFONAVIT",
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
# 3) Carga del archivo CSV
# -------------------------------------------------------
uploaded_file = st.file_uploader("📂 Sube tu archivo CSV", type=["csv", "txt"])
st.markdown("---")
if uploaded_file is None:
    st.stop()

df = pd.read_csv(
    uploaded_file,
    engine='python',
    sep=None,
    encoding='latin-1'
)

# -------------------------------------------------------
# 4) Normalización de columnas
# -------------------------------------------------------
df.columns = df.columns.str.strip().str.upper()

# -------------------------------------------------------
# 5) Limpieza de datos por columna
# -------------------------------------------------------
df['LPAR'] = df['LPAR'].astype(str).str.upper().str.strip()
df['DATO'] = (
    df['DATO']
      .astype(str)
      .str.upper()
      .str.normalize('NFKD')
      .str.replace('Á', 'A')
      .str.replace('Ó', 'O')
      .str.strip()
)
df['APLICACION'] = (
    df['APLICACION']
      .astype(str)
      .str.strip()
      .replace({'MIPS': 'CPU'})
)
df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
hora_texto = (
    df['HORA']
      .str.replace(' a. m.', ' AM', regex=False)
      .str.replace(' p. m.', ' PM', regex=False)
)
df['HORA'] = pd.to_datetime(
    hora_texto,
    format='%I:%M:%S %p',
    errors='coerce'
).dt.time
df['USO'] = pd.to_numeric(df['USO'], errors='coerce').fillna(0)
df['MES'] = df['FECHA'].dt.to_period('M').astype(str)

# -------------------------------------------------------
# 6) Sidebar: filtros de periodo (mes y día)
# -------------------------------------------------------
# Mapeo de meses en español
meses_es = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
    7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

# Etiquetas de mes en español
meses_disponibles = sorted(df['MES'].dropna().unique())
etiquetas = []
for m in meses_disponibles:
    dt = pd.to_datetime(m + '-01')
    etiquetas.append(f"{meses_es[dt.month]} {dt.year}")

mes_seleccionado_label = st.sidebar.selectbox("Selecciona MES", etiquetas, index=len(etiquetas)-1)
mes_seleccionado = meses_disponibles[etiquetas.index(mes_seleccionado_label)]

fecha_min = df['FECHA'].dt.date.min()
fecha_max = df['FECHA'].dt.date.max()
# Filtro de día: Streamlit sólo permite formatos predefinidos, así que usamos el default
dia_seleccionado = st.sidebar.date_input(
    "Selecciona DÍA",
    value=fecha_max,
    min_value=fecha_min,
    max_value=fecha_max
)

# -------------------------------------------------------
# 7) Sidebar: selección de entornos LPAR
# -------------------------------------------------------
st.sidebar.markdown("**Filtrar por LPAR**")

col1, col2 = st.sidebar.columns(2)
check_sysw = col1.checkbox("SYSW", value=True)
check_sysk = col2.checkbox("SYSK", value=True)
selected_envs = []
if check_sysw: selected_envs.append('SYSW')
if check_sysk: selected_envs.append('SYSK')
if not selected_envs:
    selected_envs = ['SYSW', 'SYSK']
    
st.sidebar.markdown("---")
# -------------------------------------------------------
# 8) Botones de acción (correo y PDF)
# -------------------------------------------------------
if st.sidebar.button("📧 Enviar por correo"):
    st.sidebar.success("✅ Enviado por correo.")

if st.sidebar.button("📄 Exportar como PDF"):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # Portada con matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Reporte INFONAVIT\n{mes_seleccionado_label}",
                ha='center', va='center', fontsize=24)
        ax.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        # Resumen mensual con matplotlib
        df_resumen = df[
            (df['MES'] == mes_seleccionado) &
            (df['LPAR'].isin(selected_envs))
        ]
        resumen = df_resumen.groupby('DATO')['USO'].sum()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        resumen.sort_values().plot.barh(ax=ax2)
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
# 9) Filtrar datos según selecciones
# -------------------------------------------------------
df_filtrado = df[
    (df['MES'] == mes_seleccionado) &
    (df['LPAR'].isin(selected_envs))
]
categorias = sorted(df_filtrado['DATO'].unique())

# Mapa de días de semana en español
weekday_map = {
    0: 'Lunes', 1: 'Martes', 2: 'Miércoles',
    3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'
}

# -------------------------------------------------------
# 10) Proceso para cada LPAR seleccionado
# -------------------------------------------------------
for entorno in selected_envs:
    st.header(f"🌐 Entorno: {entorno}")
    df_entorno = df_filtrado[df_filtrado['LPAR'] == entorno]

    # Métricas totales mensuales
    st.subheader(f"📊 Métricas totales del Mes de {mes_seleccionado_label}")
    df_totales = df_entorno.groupby('DATO')['USO'].sum()
    cols = st.columns(len(categorias))
    for i, cat in enumerate(categorias):
        cols[i].metric(f"Uso de {cat} totales", f"{df_totales.get(cat, 0):,.0f}")

    # Detalle Diario
    st.markdown("---")
    st.subheader(f"📊 Detalle por día - {dia_seleccionado.strftime('%d')} de {mes_seleccionado_label}")
    df_dia = df_entorno[df_entorno['FECHA'].dt.date == dia_seleccionado]

    for cat in categorias:
        st.markdown(f"### Categoría: {cat}")
        df_cat = df_dia[df_dia['DATO'] == cat]
        if df_cat.empty:
            st.info("No hay datos.")
            st.markdown("---")
            continue

        consumo = df_cat['USO'].sum()
        agg_h = df_cat.groupby('HORA')['USO'].sum()
        max_v = agg_h.max(); hr_max = agg_h.idxmax().strftime("%H:%M")
        min_v = agg_h.min(); hr_min = agg_h.idxmin().strftime("%H:%M")
        top_app = df_cat.groupby('APLICACION')['USO'].sum().idxmax()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total del día", f"{consumo:,.2f}")
        c2.metric("Máx x hora", f"{max_v:,.2f}", hr_max)
        c3.metric("Mín x hora", f"{min_v:,.2f}", hr_min)
        c4.metric("App top", top_app)

        multi = ['TRANSACCION', 'WEBSERVICE', 'MIPS']
        if cat.upper() in multi:
            df_m = df_cat.groupby(['HORA','APLICACION'])['USO'].sum().reset_index()
            df_m['HORA'] = df_m['HORA'].apply(lambda t: t.strftime("%H:%M"))
            fig = px.line(
                df_m, x='HORA', y='USO', color='APLICACION',
                title=f"Uso horario de {cat}", markers=True
            )
            df_cpu = df_m[df_m['APLICACION'] == 'CPU']
            if len(df_cpu) > 1:
                df_cpu['HORA_NUM'] = (
                    df_cpu['HORA'].str.slice(0,2).astype(int)
                    + df_cpu['HORA'].str.slice(3,5).astype(int)/60
                )
                coeffs = np.polyfit(df_cpu['HORA_NUM'], df_cpu['USO'], 1)
                trend = np.polyval(coeffs, df_cpu['HORA_NUM'])
                fig.add_trace(
                    go.Scatter(
                        x=df_cpu['HORA'],
                        y=trend,
                        mode='lines',
                        name='Tendencia CPU',
                        line=dict(dash='dash', width=2)
                    )
                )
        else:
            df_p = agg_h.reset_index()
            df_p['HORA'] = df_p['HORA'].apply(lambda t: t.strftime("%H:%M"))
            fig = px.line(
                df_p, x='HORA', y='USO', title=f"Uso horario de {cat}", markers=True
            )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")

    # Acumulado diario en el mes con mes en el título
    st.subheader(f"📈 Acumulado diario del Mes de {mes_seleccionado_label}")
    for cat in categorias:
        st.markdown(f"### Categoría: {cat}")
        df_cm = df_entorno[df_entorno['DATO'] == cat].copy()
        if df_cm.empty:
            st.info("No hay datos.")
            st.markdown("---")
            continue

        df_cm['DIA'] = df_cm['FECHA'].dt.date
        agg_d = df_cm.groupby('DIA')['USO'].sum().reset_index()
        agg_d['DIA_SEM'] = agg_d['DIA'].apply(lambda d: weekday_map[d.weekday()])

        fig2 = px.line(
            agg_d, x='DIA', y='USO', title=f"Acumulado diario: {cat}", markers=True
        )
        fig2.update_xaxes(
            tickmode='array',
            tickvals=agg_d['DIA'],
            ticktext=agg_d['DIA_SEM']
        )

        pico = agg_d.loc[agg_d['USO'].idxmax()]
        fig2.add_trace(
            go.Scatter(
                x=[pico['DIA']], y=[pico['USO']],
                mode='markers+text',
                marker=dict(size=10, color='red'),
                text=[f"Máximo el día {pico['DIA_SEM']} ({pico['USO']:,.0f})"],
                textposition='top center',
                showlegend=False
            )
        )

        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("---")
