
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

# Parámetros de entrada
ruta_robos = st.file_uploader("Sube shapefile de los eventos (.zip)", type=["zip"])
ruta_contorno = st.file_uploader("Opcional: Shapefile contorno/calles (.zip)", type=["zip"])
ruta_covariables = st.file_uploader("Shapefile de covariables (por polígonos)", type=["zip"])

cell_size = st.number_input("Tamaño celda rejilla (m)", min_value=100, max_value=2000, value=500, step=100)
umbral = st.slider("Umbral de probabilidad para riesgo", 0.0, 1.0, 0.7, 0.05)
mes_simulacion = st.text_input("Mes a simular (YYYY-MM)", "2019-09")
fecha_entreno_inicio = st.text_input("Inicio entrenamiento (YYYY-MM)", "2017-01")
fecha_entreno_fin = st.text_input("Fin entrenamiento (YYYY-MM)", "2019-08")
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
        return gpd.read_file(os.path.join(tmpdir, shp_files[0]))

if ruta_robos is None:
    st.warning("Sube el shapefile de puntos para continuar.")
    st.stop()

gdf = cargar_shapefile_zip(ruta_robos)
if gdf is None:
    st.stop()

gdf = gdf.to_crs(epsg=32616)
gdf["Fecha"] = gdf["Fecha"].apply(lambda x: parse(str(x), dayfirst=True, fuzzy=True) if pd.notnull(x) else pd.NaT)
gdf = gdf.dropna(subset=["Fecha"])
gdf["month"] = gdf["Fecha"].dt.to_period("M")

# Rejilla
xmin, ymin, xmax, ymax = gdf.total_bounds
cols = np.arange(xmin, xmax, cell_size)
rows = np.arange(ymin, ymax, cell_size)
polygons, cell_ids = [], []
for i, x in enumerate(cols):
    for j, y in enumerate(rows):
        polygons.append(box(x, y, x + cell_size, y + cell_size))
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
    st.error("No hay meses de entrenamiento válidos.")
    st.stop()

gdf_covars = cargar_shapefile_zip(ruta_covariables) if ruta_covariables else None
vars_disponibles = []
if gdf_covars is not None:
    gdf_covars = gdf_covars.to_crs(gdf.crs)
    vars_disponibles = gdf_covars.select_dtypes(include=[np.number]).columns.tolist()
covars_seleccionadas = st.multiselect("Selecciona covariables a usar", vars_disponibles)

if st.button("Ejecutar simulación"):

    if gdf_covars is not None and covars_seleccionadas:
        gdf_grid = gpd.sjoin(gdf_grid, gdf_covars[covars_seleccionadas + ["geometry"]], predicate="intersects", how="left")

    data = []
        for m in tqdm(train_months, desc="Generando dataset de entrenamiento"):
        df_month = gdf[gdf["month"] == m]

        # Evitar colisiones de columnas reservadas por sjoin
        for col in ['index_left', 'index_right']:
            if col in gdf_grid.columns:
                gdf_grid = gdf_grid.drop(columns=[col])
            if col in df_month.columns:
                df_month = df_month.drop(columns=[col])

        joined = gpd.sjoin(gdf_grid, df_month, predicate='contains', how='left')
        joined["label"] = joined["index_right"].notnull().astype(int)

        grouped = joined.groupby("cell_id").agg(label=("label", "max")).reset_index()
        merged = pd.merge(grouped, gdf_grid, on="cell_id")
        merged["month"] = str(m)
        data.append(merged)


    df_model = pd.concat(data, ignore_index=True)

    feature_cols = ["X", "Y"] + covars_seleccionadas
    num_total = len(df_model)
    df_model = df_model.dropna(subset=feature_cols + ["label"])
    st.write("Variables usadas:", feature_cols)
    st.write("Filas eliminadas por NaN:", num_total - len(df_model))

    for col in covars_seleccionadas:
        if df_model[col].nunique() <= 1:
            st.warning(f"Covariable '{col}' tiene un solo valor. No aporta al modelo.")

    X = df_model[feature_cols]
    y = df_model["label"]

    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    importancia_df = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values(by="importance", ascending=False)
    st.write("### Importancia de variables")
    st.dataframe(importancia_df)

    df_next = gdf_grid.copy()
    df_next = df_next.dropna(subset=feature_cols)
    X_pred = df_next[feature_cols]
    probs = model.predict_proba(X_pred)[:, 1]
    df_next["predicted_prob"] = probs
    df_next["predicted_risk"] = (probs >= umbral).astype(int)

    gdf_contorno = cargar_shapefile_zip(ruta_contorno) if ruta_contorno else None
    if gdf_contorno is not None:
        gdf_contorno = gdf_contorno.to_crs(gdf.crs)

    df_test_month = gdf[gdf["month"] == mes_sim]

    st.write(f"Celdas con riesgo predicho: {df_next['predicted_risk'].sum()}")
    fig, ax = plt.subplots(figsize=(12, 10))
    if gdf_contorno is not None:
        gdf_contorno.plot(ax=ax, facecolor="none", edgecolor="gray", alpha=0.6)
    df_next[df_next["predicted_risk"] == 1].plot(ax=ax, facecolor="red", edgecolor="darkred", alpha=0.4, linewidth=0.7)
    df_test_month.plot(ax=ax, color="black", markersize=8, alpha=0.7)
    plt.title(titulo_mapa)
    plt.axis("off")
    st.pyplot(fig)

    joined_test = gpd.sjoin(df_next, df_test_month, predicate='contains', how='left')
    joined_test["actual_label"] = joined_test["index_right"].notnull().astype(int)
    evaluated = joined_test.groupby("cell_id").agg(predicted=("predicted_risk", "max"), actual=("actual_label", "max")).reset_index()
    y_true, y_pred = evaluated["actual"], evaluated["predicted"]
    st.write("### Evaluación por celda:")
    st.write(f"Precision: {precision_score(y_true, y_pred):.2f}")
    st.write(f"Recall:    {recall_score(y_true, y_pred):.2f}")
    st.write(f"F1-score:  {f1_score(y_true, y_pred):.2f}")

    def to_geojson_bytes(gdf):
        allowed_types = (int, float, str, bool, type(None), np.number)
        cols_validas = [col for col in gdf.columns if col == 'geometry' or gdf[col].map(lambda x: isinstance(x, allowed_types)).all()]
        return gdf[cols_validas].to_json().encode('utf-8')

    def to_shapefile_bytes(gdf):
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = os.path.join(tmpdir, "prediccion.shp")
            gdf.to_file(shp_path)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                    f = os.path.join(tmpdir, f"prediccion{ext}")
                    if os.path.exists(f):
                        zf.write(f, arcname=f"prediccion{ext}")
            return zip_buffer.getvalue()

    geojson_bytes = to_geojson_bytes(df_next)
    st.download_button("Descargar GeoJSON", geojson_bytes, file_name="prediccion.geojson", mime="application/geo+json")

    shapefile_bytes = to_shapefile_bytes(df_next)
    st.download_button("Descargar Shapefile (zip)", shapefile_bytes, file_name="prediccion.zip", mime="application/zip")
