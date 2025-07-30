import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import io
import os
import tempfile
import zipfile
from dateutil.parser import parse

st.title("Simulación de riesgo espacial de eventos delictivos")

# Parámetros configurables
ruta_robos = st.file_uploader("Sube shapefile de los eventos (.shp + .shx + .dbf + ... en zip)", type=["zip"])
ruta_contorno = st.file_uploader("Opcional: Subir shapefile contorno/calles (.zip)", type=["zip"])
ruta_covariables = st.file_uploader("Shapefile de covariables por sección censal (.zip)", type=["zip"])
cell_size = st.number_input("Tamaño celda rejilla (metros)", min_value=100, max_value=2000, value=500, step=100)
umbral = st.slider("Umbral probabilidad para riesgo", 0.0, 1.0, 0.7, 0.05)
mes_simulacion = st.text_input("Mes a simular (formato YYYY-MM)", "2019-09")
fecha_entreno_inicio = st.text_input("Fecha inicio entrenamiento (YYYY-MM)", "2017-01")
fecha_entreno_fin = st.text_input("Fecha fin entrenamiento (YYYY-MM)", "2019-08")

titulo_mapa = st.text_input("Título del mapa", f"Riesgo predicho vs hurtos reales - {mes_simulacion}")

def cargar_shapefile_zip(zip_file):
    if zip_file is None:
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall(tmpdir)
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if not shp_files:
                st.error("No se encontró archivo .shp en el zip.")
                return None
            return gpd.read_file(os.path.join(tmpdir, shp_files[0]))

if ruta_robos is None:
    st.warning("Sube el shapefile de puntos para continuar.")
    st.stop()

gdf = cargar_shapefile_zip(ruta_robos)
if gdf is None:
    st.stop()

gdf = gdf.to_crs(epsg=32616)  # UTM metros

def parse_fecha_segura(fecha):
    try:
        return parse(str(fecha), dayfirst=True, fuzzy=True)
    except:
        return pd.NaT

gdf["Fecha"] = gdf["Fecha"].apply(parse_fecha_segura)

errores = gdf[gdf["Fecha"].isna()]
if not errores.empty:
    st.warning(f"No se pudieron reconocer {len(errores)} fechas y serán descartadas.")

gdf = gdf.dropna(subset=["Fecha"])
gdf["month"] = gdf["Fecha"].dt.to_period("M")

xmin, ymin, xmax, ymax = gdf.total_bounds
cols = list(np.arange(xmin, xmax, cell_size))
rows = list(np.arange(ymin, ymax, cell_size))
polygons = []
cell_ids = []
for i, x in enumerate(cols):
    for j, y in enumerate(rows):
        poly = box(x, y, x + cell_size, y + cell_size)
        polygons.append(poly)
        cell_ids.append(f"{i}_{j}")

gdf_grid = gpd.GeoDataFrame({'cell_id': cell_ids}, geometry=polygons, crs=gdf.crs)
gdf_grid["X"] = gdf_grid.geometry.centroid.x
gdf_grid["Y"] = gdf_grid.geometry.centroid.y

# Covariables por sección censal
gdf_covars = None
selected_vars = []
if ruta_covariables is not None:
    gdf_covars = cargar_shapefile_zip(ruta_covariables)
    if gdf_covars is not None:
        gdf_covars = gdf_covars.to_crs(gdf_grid.crs)
        posibles_vars = [col for col in gdf_covars.columns if np.issubdtype(gdf_covars[col].dtype, np.number) and col != 'geometry']
        selected_vars = st.multiselect("Selecciona variables para usar como covariables", posibles_vars)

mes_entreno_inicio = pd.Period(fecha_entreno_inicio, freq="M")
mes_entreno_fin = pd.Period(fecha_entreno_fin, freq="M")
mes_sim = pd.Period(mes_simulacion, freq="M")

train_months = [m for m in sorted(gdf["month"].unique()) if mes_entreno_inicio <= m <= mes_entreno_fin]

if mes_sim not in gdf["month"].unique():
    st.error(f"Mes de simulación {mes_simulacion} no está en los datos.")
    st.stop()

if len(train_months) == 0:
    st.error("No hay meses de entrenamiento válidos en el rango seleccionado.")
    st.stop()

st.write(f"Entrenando con meses desde {fecha_entreno_inicio} hasta {fecha_entreno_fin}")
st.write(f"Simulando mes: {mes_simulacion}")

if st.button("Ejecutar simulación"):

    data = []
    for m in tqdm(train_months, desc="Generando dataset de entrenamiento"):
        df_month = gdf[gdf["month"] == m]
        joined = gpd.sjoin(gdf_grid, df_month, predicate='contains', how='left')
        joined["label"] = joined["index_right"].notnull().astype(int)
        grouped = joined.groupby("cell_id").agg(label=("label", "max")).reset_index()
        merged = pd.merge(grouped, gdf_grid, on="cell_id")

        if gdf_covars is not None and selected_vars:
            if not isinstance(merged, gpd.GeoDataFrame):
                merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=gdf_grid.crs)

            inter = gpd.sjoin(merged, gdf_covars[selected_vars + ['geometry']], how="left", predicate='intersects')
            for var in selected_vars:
                merged[var] = inter.groupby("cell_id")[var].transform("mean")

        merged["month"] = str(m)
        data.append(merged)

    df_model = pd.concat(data, ignore_index=True)

    features = ["X", "Y"] + selected_vars
    df_model = df_model.dropna(subset=features)

    X = df_model[features]
    y = df_model["label"]

    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X, y)

    df_next = gdf_grid.copy()

    if gdf_covars is not None and selected_vars:
        if not isinstance(df_next, gpd.GeoDataFrame):
            df_next = gpd.GeoDataFrame(df_next, geometry='geometry', crs=gdf_grid.crs)

        inter_pred = gpd.sjoin(df_next, gdf_covars[selected_vars + ['geometry']], how="left", predicate='intersects')
        for var in selected_vars:
            df_next[var] = inter_pred.groupby("cell_id")[var].transform("mean")

    X_pred = df_next[features].fillna(0)

    probs = model.predict_proba(X_pred)[:, 1]
    df_next["predicted_prob"] = probs
    df_next["predicted_risk"] = (probs >= umbral).astype(int)

    st.write("### Importancia de variables")
    importances = model.feature_importances_
    for name, val in zip(features, importances):
        st.write(f"{name}: {val:.3f}")

    gdf_contorno = None
    if ruta_contorno is not None:
        gdf_contorno = cargar_shapefile_zip(ruta_contorno)
        if gdf_contorno is not None:
            gdf_contorno = gdf_contorno.to_crs(gdf.crs)

    df_test_month = gdf[gdf["month"] == mes_sim]

    st.write(f"Celdas con riesgo predicho == 1: {df_next['predicted_risk'].sum()}")

    fig, ax = plt.subplots(figsize=(12, 10))

    if gdf_contorno is not None:
        gdf_contorno.plot(ax=ax, facecolor="none", edgecolor="gray", alpha=0.6)

    df_next_risk = df_next[df_next["predicted_risk"] == 1]
    if not df_next_risk.empty:
        df_next_risk.plot(ax=ax, facecolor="red", edgecolor="darkred", alpha=0.4, linewidth=0.7, label="Riesgo alto")
    else:
        st.warning("No se detectaron celdas con riesgo alto según el umbral.")

    if not df_test_month.empty:
        df_test_month.plot(ax=ax, color="black", markersize=8, alpha=0.7, label="Eventos reales")
    else:
        st.warning("No hay eventos reales para el mes seleccionado.")

    plt.legend()
    plt.title(titulo_mapa)
    plt.axis("off")
    st.pyplot(fig)

    joined_test = gpd.sjoin(df_next, df_test_month, predicate='contains', how='left')
    joined_test["actual_label"] = joined_test["index_right"].notnull().astype(int)
    evaluated = joined_test.groupby("cell_id").agg(
