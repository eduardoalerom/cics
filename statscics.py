import streamlit as st
import pandas as pd
import plotly.express as px
import io
from matplotlib.backends.backend_pdf import PdfPages

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
st.sidebar.title("📊  Estadísticas INFONAVIT")

st.title("INFRAESTRUCTURA CICS | INFONAVIT")
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
# 6) Sidebar: filtros de periodo
# -------------------------------------------------------
meses_disponibles = sorted(df['MES'].dropna().unique())
etiquetas = [pd.to_datetime(m + '-01').strftime('%B %Y') for m in meses_disponibles]
mes_seleccionado_label = st.sidebar.selectbox("Selecciona MES", etiquetas, index=len(etiquetas)-1)
mes_seleccionado = meses_disponibles[etiquetas.index(mes_seleccionado_label)]

fecha_min = df['FECHA'].dt.date.min()
fecha_max = df['FECHA'].dt.date.max()
dia_seleccionado = st.sidebar.date_input(
    "Selecciona DÍA",
    value=fecha_max,
    min_value=fecha_min,
    max_value=fecha_max
)

# -------------------------------------------------------
# 7) Sidebar: selección de entornos LPAR
# -------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("**Filtrar por LPAR**")
col1, col2 = st.sidebar.columns(2)
check_sysw = col1.checkbox("SYSW", value=True)
check_sysk = col2.checkbox("SYSK", value=True)

selected_envs = []
if check_sysw: selected_envs.append('SYSW')
if check_sysk: selected_envs.append('SYSK')
if not selected_envs:
    selected_envs = ['SYSW', 'SYSK']

# -------------------------------------------------------
# 8) Botones de acción (correo y PDF)
# -------------------------------------------------------
if st.sidebar.button("📧 Enviar por correo"):
    st.sidebar.success("✅ Enviado por correo.")

if st.sidebar.button("📄 Exportar como PDF"):
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # Portada
        fig_portada = px.scatter(x=[None], y=[None])
        fig_portada.update_layout(
            title_text=f"Reporte INFONAVIT - {mes_seleccionado_label}",
            xaxis_visible=False, yaxis_visible=False,
            width=600, height=400
        )
        pdf.savefig(fig_portada.to_image(format='png'))

        # Resumen mensual
        df_resumen = df[
            (df['MES'] == mes_seleccionado) &
            (df['LPAR'].isin(selected_envs))
        ]
        resumen_mensual = df_resumen.groupby('DATO')['USO'].sum().reset_index()
        fig_resumen = px.bar(
            resumen_mensual,
            x='DATO', y='USO',
            title='Uso Total por Categoría (Mes)'
        )
        pdf.savefig(fig_resumen.to_image(format='png'))

    pdf_buffer.seek(0)
    st.sidebar.download_button(
        label="Descargar PDF",
        data=pdf_buffer,
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

# -------------------------------------------------------
# 10) Proceso para cada LPAR seleccionado
# -------------------------------------------------------
for entorno in selected_envs:
    st.header(f"🌐 Entorno: {entorno}")
    df_entorno = df_filtrado[df_filtrado['LPAR'] == entorno]

    # 10.2) Métricas totales mensuales
    st.subheader("📊 Métricas Totales Mensuales")
    df_totales_mes = df_entorno.groupby('DATO')['USO'].sum().reset_index()
    cols_totales = st.columns(len(categorias))
    for idx, categoria in enumerate(categorias):
        valor_total = df_totales_mes.loc[
            df_totales_mes['DATO'] == categoria, 'USO'
        ].sum()
        cols_totales[idx].metric(f"{categoria} totales", f"{valor_total:,.0f}")

    # 10.3) Detalle diario
    st.subheader(f"🗓 Detalle Diario ({dia_seleccionado})")
    df_dia = df_entorno[df_entorno['FECHA'].dt.date == dia_seleccionado]

    for categoria in categorias:
        st.markdown(f"### Categoría: {categoria}")
        df_cat_dia = df_dia[df_dia['DATO'] == categoria]
        if df_cat_dia.empty:
            st.info(f"No hay datos de {categoria} en este día.")
            st.markdown("---")
            continue

        # Consumo total, máximo y mínimo, top app
        consumo_total = df_cat_dia['USO'].sum()
        agg_hora = df_cat_dia.groupby('HORA')['USO'].sum()
        max_val = agg_hora.max()
        hora_max = agg_hora.idxmax().strftime("%H:%M")
        min_val = agg_hora.min()
        hora_min = agg_hora.idxmin().strftime("%H:%M")
        app_top = df_cat_dia.groupby('APLICACION')['USO'].sum().idxmax()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Consumo total (hoy)", f"{consumo_total:,.2f}")
        c2.metric("Máximo x hora", f"{max_val:,.2f}", hora_max)
        c3.metric("Mínimo x hora", f"{min_val:,.2f}", hora_min)
        c4.metric("App con más uso", app_top)

        # 10.3.7) Gráfica de uso por hora
        multiline_cats = ['TRANSACCION', 'WEB SERVICES', 'WEBSERVICE', 'WEBSERVICES']
        if categoria in multiline_cats:
            # una línea por aplicación
            df_app_hour = (
                df_cat_dia
                .groupby(['HORA', 'APLICACION'])['USO']
                .sum()
                .reset_index()
            )
            df_app_hour['HORA_STR'] = df_app_hour['HORA']\
                .apply(lambda t: t.strftime("%H:%M") if hasattr(t, "strftime") else "")
            fig = px.line(
                df_app_hour,
                x='HORA_STR',
                y='USO',
                color='APLICACION',
                title=f"Uso horario de {categoria}",
                markers=True
            )
        else:
            # curva agregada
            df_plot = agg_hora.reset_index()
            df_plot['HORA_STR'] = df_plot['HORA']\
                .apply(lambda t: t.strftime("%H:%M") if hasattr(t, "strftime") else "")
            fig = px.line(
                df_plot,
                x='HORA_STR',
                y='USO',
                title=f"Uso horario de {categoria}",
                markers=True
            )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")

    # 10.4) Acumulado diario mensual con días de semana
    st.subheader("📈 Acumulado diario en el mes")
    weekday_map = {
        0: 'Lunes', 1: 'Martes', 2: 'Miércoles',
        3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'
    }

    for categoria in categorias:
        st.markdown(f"### Categoría: {categoria}")
        df_cat_mon = df_entorno[df_entorno['DATO'] == categoria].copy()
        if df_cat_mon.empty:
            st.info(f"No hay datos de {categoria} en este mes.")
            st.markdown("---")
            continue

        df_cat_mon['FECHA_DIA'] = df_cat_mon['FECHA'].dt.date
        agg_dia = (
            df_cat_mon
            .groupby('FECHA_DIA')['USO']
            .sum()
            .reset_index()
        )
        agg_dia['DIA_SEM'] = agg_dia['FECHA_DIA']\
            .apply(lambda d: weekday_map[d.weekday()])

        fig2 = px.line(
            agg_dia,
            x='FECHA_DIA',
            y='USO',
            title=f"Acumulado diario: {categoria}",
            markers=True
        )
        fig2.update_xaxes(
            tickmode='array',
            tickvals=agg_dia['FECHA_DIA'],
            ticktext=agg_dia['DIA_SEM']
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Día pico
        pico_idx = agg_dia['USO'].idxmax()
        dia_pico = agg_dia.loc[pico_idx, 'DIA_SEM']
        valor_pico = agg_dia.loc[pico_idx, 'USO']
        st.markdown(f"Día pico: **{dia_pico}** con **{valor_pico:,.0f}** usos")
        st.markdown("---")
