import streamlit as st
import pandas as pd
import plotly.express as px
import io
#from matplotlib.backends.backend_pdf import PdfPages

# Configuración de la página
st.set_page_config(
    page_title="Estadísticas INFONAVIT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar: logo y título
st.sidebar.image("logo_infonavit.png", width=100)
st.sidebar.title("📊  Estadísticas INFONAVIT")

# Cabecera y descripción
st.title("INFRAESTRUCTURA CICS | INFONAVIT")
st.markdown(
    """
    El presente Dashboard nos permite transformar volúmenes masivos de información transaccional
    en gráficos interactivos y reportes de alto valor estratégico, visualizando comportamientos y
    tendencias a lo largo de periodos de tiempo.
    """
)

# Carga y preprocesamiento de datos
df_file = st.file_uploader("📂 Sube tu CSV", type=["csv", "txt"])
if not df_file:
    st.stop()

df = pd.read_csv(df_file, engine='python', sep=None, encoding='latin-1')
df.columns = df.columns.str.strip().str.upper()
df['LPAR'] = df['LPAR'].astype(str).str.upper().str.strip()
df['DATO'] = (
    df['DATO'].astype(str)
      .str.upper()
      .str.normalize('NFKD')
      .str.replace('Á','A')
      .str.replace('Ó','O')
      .str.strip()
)
df['APLICACION'] = df['APLICACION'].astype(str).str.strip()
df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
h = df['HORA'].str.replace(' a. m.', ' AM', regex=False).str.replace(' p. m.', ' PM', regex=False)
df['HORA'] = pd.to_datetime(h, format='%I:%M:%S %p', errors='coerce').dt.time
df['USO'] = pd.to_numeric(df['USO'], errors='coerce').fillna(0)
df['MES'] = df['FECHA'].dt.to_period('M').astype(str)

# Sidebar: filtros de periodo y fecha
periods = sorted(df['MES'].dropna().unique())
labels = [pd.to_datetime(p + '-01').strftime('%B %Y') for p in periods]
sel_label = st.sidebar.selectbox("Selecciona MES", labels, index=len(labels)-1)
sel_mes = periods[labels.index(sel_label)]

day_min = df['FECHA'].dt.date.min()
day_max = df['FECHA'].dt.date.max()
sel_day = st.sidebar.date_input("Selecciona Día", value=day_max, min_value=day_min, max_value=day_max)

# Sidebar: selección de entornos y acciones
st.sidebar.markdown("---")
st.sidebar.markdown("**Filtrar Entornos LPAR**")
env1, env2 = st.sidebar.columns(2)
sysw = env1.checkbox("SYSW", value=True)
sysk = env2.checkbox("SYSK", value=True)
selected_envs = [e for e, f in zip(['SYSW','SYSK'], [sysw, sysk]) if f] or ['SYSW','SYSK']

# Botones de acción
if st.sidebar.button("📧 Enviar por correo"):
    # TODO: lógica de envío de correo
    st.sidebar.success("✅ Gráficas enviadas por correo.")
if st.sidebar.button("📄 Exportar como PDF"):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # Título
        fig = px.scatter(x=[None], y=[None])
        fig.update_layout(
            title_text=f"Reporte INFONAVIT - {sel_label}",
            xaxis_visible=False, yaxis_visible=False,
            width=600, height=400)
        pdf.savefig(fig.to_image(format='png'))
        # Resumen por categoría
        summary = df[(df['MES']==sel_mes) & (df['LPAR'].isin(selected_envs))] \
                   .groupby('DATO')['USO'].sum().reset_index()
        fig2 = px.bar(summary, x='DATO', y='USO', title='Uso Total por Categoría')
        pdf.savefig(fig2.to_image(format='png'))
    buffer.seek(0)
    st.sidebar.download_button(
        label="Descargar PDF",
        data=buffer,
        file_name=f"report_{sel_mes}.pdf",
        mime="application/pdf"
    )

# Filtrar datos
data_filtered = df[(df['MES']==sel_mes) & (df['LPAR'].isin(selected_envs))]
categories = sorted(data_filtered['DATO'].unique())

# Renderizado por entorno
for env in selected_envs:
    st.header(f"🌐 Entorno: {env}")
    df_env = data_filtered[data_filtered['LPAR']==env]

    # Métricas totales
    totals = df_env.groupby('DATO')['USO'].sum().reset_index()
    metric_cols = st.columns(len(categories))
    for col, cat in zip(metric_cols, categories):
        value = totals.loc[totals['DATO']==cat, 'USO'].sum()
        col.metric(cat.upper() + ' TOTALES', f"{value:,.0f}")

    # Gráficas sumarias dinámicas
    st.subheader("🔹 Gráficas sumarias por Categoría")
    sum_cols = st.columns(len(categories))
    for col_obj, cat in zip(sum_cols, categories):
        df_cat = df_env[df_env['DATO']==cat]
        if not df_cat.empty:
            agg = df_cat.groupby('HORA')['USO'].sum().reset_index()
            agg['HORA_STR'] = agg['HORA'].apply(lambda t: t.strftime('%H:%M') if hasattr(t,'strftime') else '')
            fig = px.line(agg, x='HORA_STR', y='USO', title=f"{cat} - Uso por hora", markers=True)
            col_obj.plotly_chart(fig, use_container_width=True)
        else:
            col_obj.info(f"No hay datos de {cat}")

    # Top por Aplicación
    st.subheader("🔹 Top por Aplicación")
    for cat in categories:
        df_cat = df_env[df_env['DATO']==cat]
        if not df_cat.empty:
            usage_app = df_cat.groupby('APLICACION')['USO'].sum().reset_index()
            fig = px.bar(usage_app, x='APLICACION', y='USO', title=f"{cat} por Aplicación")
            st.plotly_chart(fig, use_container_width=True)
            max_app = usage_app.loc[usage_app['USO'].idxmax()]
            st.markdown(f"**Máximo**: {max_app['APLICACION']} ({max_app['USO']:,.0f} usos)")
        st.markdown('---')

    # Detalle por Categoría
    st.subheader('🔸 Uso diario vs mensual acumulado')
    for cat in categories:
        st.markdown(f"### {cat}")
        # Diario
        df_day = df_env[df_env['FECHA'].dt.date==sel_day]
        if not df_day[df_day['DATO']==cat].empty:
            df_cat_day = df_day[df_day['DATO']==cat]
            agg_day = df_cat_day.groupby('HORA')['USO'].sum().reset_index()
            agg_day['HORA_STR'] = agg_day['HORA'].apply(lambda t: t.strftime('%H:%M'))
            fig = px.line(agg_day, x='HORA_STR', y='USO', title=f"{cat}: uso por hora - {sel_day}", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            pico = agg_day.loc[agg_day['USO'].idxmax()]
            st.markdown(f"Hora pico: {pico['HORA_STR']} ({pico['USO']:,.0f} usos)")
        else:
            st.info(f"No hay {cat} en {sel_day}.")

        # Mensual acumulado
        df_cat_mon = df_env[df_env['DATO']==cat]
        if not df_cat_mon.empty:
            agg_mon = df_cat_mon.groupby(df_cat_mon['FECHA'].dt.day)['USO'].sum().reset_index().rename(columns={'FECHA':'DIA'})
            fig = px.line(agg_mon, x='DIA', y='USO', title=f"{cat}: acumulado diario - {sel_label}", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            day_pico = agg_mon.loc[agg_mon['USO'].idxmax()]
            st.markdown(f"Día pico: {int(day_pico['DIA'])} ({day_pico['USO']:,.0f} usos)")
        else:
            st.info(f"No hay {cat} en mes {sel_label}.")
        st.markdown('---')
