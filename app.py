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
ruta_covars = st.file_uploader("Opcional: Subir shapefile de covariables poligonales (.zip)", type=["zip"])
ruta_contorno = st.file_uploader("Opcional: Subir shapefile contorno/calles (.zip)", type=["zip"])

cell_size = st.number_input("Tamaño celda rejilla (metros)", min_value=100, max_value=2000, value=500, step=100)
umbral = st.slider("Umbral probabilidad para riesgo", 0.0, 1.0, 0.7, 0.05)
mes_simulacion = st.text_input("Mes a simular (formato YYYY-MM)", "2019-09")
fecha_entreno_inicio = st.text_input("Fecha inicio entrenamiento (YYYY-MM)", "2017-01")
fecha_entreno_fin = st.text_input("Fecha fin entrenamiento (YYYY-MM)", "2019-08")

# Nuevo input para título personalizado
titulo_mapa = st.text_input("Título del mapa", f"Riesgo predicho vs hurtos reales - {mes_simulacion}")

def cargar_shapefile_zip(zip_file):
    if zip_file is None:
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        z = zipfile.ZipFile(zip_file)
        z.extractall(tmpdir)
        shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
        if not shp_files:
            st.error("No se encontró archivo .shp en el zip.")
            return None
        gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
    return gdf

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

# Carga covariables y selecciona variables numéricas para multiselect
gdf_covars = None
selected_vars = []
if ruta_covars is not None:
    gdf_covars = cargar_shapefile_zip(ruta_covars)
    if gdf_covars is not None:
        gdf_covars = gdf_covars.to_crs(gdf.crs)
        numeric_cols = gdf_covars.select_dtypes(include=[np.number]).columns.tolist()
        if 'geometry' in numeric_cols:
            numeric_cols.remove('geometry')
        selected_vars = st.multiselect("Selecciona variables para covariables (polígonos)", numeric_cols)
    else:
        st.warning("No se pudo cargar correctamente el shapefile de covariables.")

st.write(f"Entrenando con meses desde {fecha_entreno_inicio} hasta {fecha_entreno_fin}")
st.write(f"Simulando mes: {mes_simulacion}")

if st.button("Ejecutar simulación"):

    data = []
    for m in tqdm(train_months, desc="Generando dataset de entrenamiento"):
        df_month = gdf[gdf["month"] == m]
        # Join espacial para etiquetas
        joined = gpd.sjoin(gdf_grid, df_month, predicate='contains', how='left')
        joined["label"] = joined["index_right"].notnull().astype(int)
        grouped = joined.groupby("cell_id").agg(label=("label", "max")).reset_index()
        merged = pd.merge(grouped, gdf_grid, on="cell_id")
        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf.crs)

        # Si hay covariables y variables seleccionadas, unir por intersección espacial (sjoin)
        if gdf_covars is not None and selected_vars:
            inter = gpd.sjoin(merged, gdf_covars[['geometry'] + selected_vars], how="left", predicate='intersects')
            inter = inter.drop(columns=['index_right'])
            # Dejar una fila por celda_id con covariables promedio (en caso de múltiples intersecciones)
            agg_dict = {var: "mean" for var in selected_vars}
            inter_agg = inter.groupby("cell_id").agg(agg_dict).reset_index()
            # Merge con merged para añadir covariables
            merged = pd.merge(merged.drop(columns=selected_vars, errors='ignore'), inter_agg, on='cell_id', how='left')
            merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf.crs)

        merged["month"] = str(m)
        data.append(merged)

    df_model = pd.concat(data, ignore_index=True)

    st.write("Columnas en df_model:", df_model.columns.tolist())
    st.write("Variables seleccionadas:", selected_vars)

    # Sólo usar variables que estén en df_model para evitar KeyError
    base_vars = ["X", "Y"]
    vars_disponibles = [v for v in selected_vars if v in df_model.columns] if selected_vars else []
    feature_vars = base_vars + vars_disponibles

    if len(feature_vars) == 0:
        st.error("No hay variables predictoras disponibles para entrenar el modelo.")
        st.stop()

    X = df_model[feature_vars].fillna(0)
    y = df_model["label"]

    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X, y)

    # Preparar datos para predicción
    df_next = gdf_grid.copy()

    if gdf_covars is not None and selected_vars:
        inter_pred = gpd.sjoin(df_next, gdf_covars[['geometry'] + selected_vars], how="left", predicate='intersects')
        inter_pred = inter_pred.drop(columns=['index_right'])
        agg_dict = {var: "mean" for var in selected_vars}
        inter_agg_pred = inter_pred.groupby("cell_id").agg(agg_dict).reset_index()
        df_next = pd.merge(df_next.drop(columns=selected_vars, errors='ignore'), inter_agg_pred, on='cell_id', how='left')
        df_next = gpd.GeoDataFrame(df_next, geometry="geometry", crs=gdf.crs)

    X_pred = df_next[feature_vars].fillna(0)

    probs = model.predict_proba(X_pred)[:, 1]
    df_next["predicted_prob"] = probs
    df_next["predicted_risk"] = (probs >= umbral).astype(int)

    gdf_contorno = None
    if ruta_contorno is not None:
        gdf_contorno = cargar_shapefile_zip(ruta_contorno)
        if gdf_contorno is not None:
            gdf_contorno = gdf_contorno.to_crs(gdf.crs)

    df_test_month = gdf[gdf["month"] == mes_sim]

    st.write(f"Celdas con riesgo predicho == 1: {df_next['predicted_risk'].sum()}")

    # Plot mapa
    fig, ax = plt.subplots(figsize=(10, 10))
    df_next.plot(column="predicted_risk", cmap="Reds", legend=True, alpha=0.6, ax=ax)
    df_test_month.plot(ax=ax, color="blue", markersize=10, label="Eventos reales")
    if gdf_contorno is not None:
        gdf_contorno.boundary.plot(ax=ax, color="black", linewidth=1)
    plt.title(titulo_mapa)
    plt.legend()
    plt.axis("off")
    st.pyplot(fig)

    # Métricas de evaluación en entrenamiento
    y_pred_train = model.predict(X)
    st.write("### Métricas de evaluación en datos de entrenamiento:")
    st.write(f"Precisión: {precision_score(y, y_pred_train):.3f}")
    st.write(f"Recall: {recall_score(y, y_pred_train):.3f}")
    st.write(f"F1 score: {f1_score(y, y_pred_train):.3f}")

    # Importancia variables
    importancias = model.feature_importances_
    df_importance = pd.DataFrame({"variable": feature_vars, "importancia": importancias}).sort_values(by="importancia", ascending=False)
    st.write("Importancia de variables:")
    st.dataframe(df_importance)

