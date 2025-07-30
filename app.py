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

# Carga de archivos
ruta_robos = st.file_uploader("Sube shapefile de eventos (.zip)", type=["zip"])
ruta_contorno = st.file_uploader("Opcional: shapefile de contorno", type=["zip"])
ruta_covariables = st.file_uploader("Opcional: shapefile de covariables", type=["zip"])

# Parámetros
cell_size = st.number_input("Tamaño celda (m)", 100, 2000, 500, 100)
umbral = st.slider("Umbral de riesgo", 0.0, 1.0, 0.7, 0.05)
mes_simulacion = st.text_input("Mes a simular (YYYY-MM)", "2019-09")
fecha_entreno_inicio = st.text_input("Inicio entrenamiento (YYYY-MM)", "2017-01")
fecha_entreno_fin = st.text_input("Fin entrenamiento (YYYY-MM)", "2019-08")
titulo_mapa = st.text_input("Título del mapa", f"Predicción {mes_simulacion}")

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

def parse_fecha_segura(fecha):
    try:
        return parse(str(fecha), dayfirst=True, fuzzy=True)
    except:
        return pd.NaT

gdf["Fecha"] = gdf["Fecha"].apply(parse_fecha_segura)
gdf = gdf.dropna(subset=["Fecha"])
gdf["month"] = gdf["Fecha"].dt.to_period("M")

# Crear rejilla
xmin, ymin, xmax, ymax = gdf.total_bounds
cols = list(np.arange(xmin, xmax, cell_size))
rows = list(np.arange(ymin, ymax, cell_size))
polygons, cell_ids = [], []

for i, x in enumerate(cols):
    for j, y in enumerate(rows):
        polygons.append(box(x, y, x + cell_size, y + cell_size))
        cell_ids.append(f"{i}_{j}")

gdf_grid = gpd.GeoDataFrame({'cell_id': cell_ids}, geometry=polygons, crs=gdf.crs)
gdf_grid["X"] = gdf_grid.geometry.centroid.x
gdf_grid["Y"] = gdf_grid.geometry.centroid.y

# Cargar covariables
gdf_covars, selected_vars = None, []
if ruta_covariables:
    gdf_covars = cargar_shapefile_zip(ruta_covariables)
    if gdf_covars is not None:
        gdf_covars = gdf_covars.to_crs(gdf.crs)
        numeric_cols = gdf_covars.select_dtypes(include=np.number).columns.tolist()
        selected_vars = st.multiselect("Selecciona covariables", numeric_cols)

# Fechas
mes_entreno_inicio = pd.Period(fecha_entreno_inicio, freq="M")
mes_entreno_fin = pd.Period(fecha_entreno_fin, freq="M")
mes_sim = pd.Period(mes_simulacion, freq="M")
train_months = [m for m in sorted(gdf["month"].unique()) if mes_entreno_inicio <= m <= mes_entreno_fin]

if mes_sim not in gdf["month"].unique():
    st.error(f"Mes de simulación {mes_simulacion} no está en los datos.")
    st.stop()

if st.button("Ejecutar simulación"):
    data = []

    for m in tqdm(train_months, desc="Entrenando"):
        df_month = gdf[gdf["month"] == m]

        for col in ['index_left', 'index_right']:
            if col in gdf_grid.columns:
                gdf_grid = gdf_grid.drop(columns=[col])
            if col in df_month.columns:
                df_month = df_month.drop(columns=[col])

        joined = gpd.sjoin(gdf_grid, df_month, predicate='contains', how='left')
        joined["label"] = joined["index_right"].notnull().astype(int)

        grouped = joined.groupby("cell_id").agg(label=("label", "max")).reset_index()
        merged = pd.merge(grouped, gdf_grid, on="cell_id")

        if gdf_covars is not None and selected_vars:
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
        inter = gpd.sjoin(df_next, gdf_covars[selected_vars + ['geometry']], how="left", predicate='intersects')
        for var in selected_vars:
            df_next[var] = inter.groupby("cell_id")[var].transform("mean")
        df_next = df_next.dropna(subset=features)

    X_pred = df_next[features]
    probs = model.predict_proba(X_pred)[:, 1]
    df_next["predicted_prob"] = probs
    df_next["predicted_risk"] = (probs >= umbral).astype(int)

    # Evaluación
    df_test = gdf[gdf["month"] == mes_sim]
    joined_test = gpd.sjoin(df_next, df_test, predicate='contains', how='left')
    joined_test["actual"] = joined_test["index_right"].notnull().astype(int)
    evaluated = joined_test.groupby("cell_id").agg(
        predicted=("predicted_risk", "max"),
        actual=("actual", "max")
    ).reset_index()

    precision = precision_score(evaluated["actual"], evaluated["predicted"])
    recall = recall_score(evaluated["actual"], evaluated["predicted"])
    f1 = f1_score(evaluated["actual"], evaluated["predicted"])

    st.write("### Métricas de evaluación")
    st.write(f"Precisión: {precision:.2f}")
    st.write(f"Recall:    {recall:.2f}")
    st.write(f"F1-score:  {f1:.2f}")

    st.write("### Importancia de variables")
    importances = model.feature_importances_
    for f, i in zip(features, importances):
        st.write(f"- {f}: {i:.3f}")

    # Mapa
    fig, ax = plt.subplots(figsize=(10, 10))
    df_next[df_next["predicted_risk"] == 1].plot(ax=ax, color="red", alpha=0.5, edgecolor="darkred", label="Riesgo alto")
    if not df_test.empty:
        df_test.plot(ax=ax, color="black", markersize=5, label="Eventos reales")
    if ruta_contorno:
        gdf_contorno = cargar_shapefile_zip(ruta_contorno)
        if gdf_contorno is not None:
            gdf_contorno.to_crs(gdf.crs).plot(ax=ax, edgecolor="gray", facecolor="none", alpha=0.4)
    plt.legend()
    plt.title(titulo_mapa)
    plt.axis("off")
    st.pyplot(fig)

    # Exportar
    def to_geojson_bytes(gdf):
        df = gdf.drop(columns=[col for col in gdf.columns if isinstance(gdf[col].dtype, np.dtype) and 'object' in str(gdf[col].dtype) and col != 'geometry'], errors='ignore')
        return df.to_json().encode("utf-8")

    def to_shapefile_bytes(gdf):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "export.shp")
            gdf.to_file(path)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as zf:
                for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                    f = path.replace(".shp", ext)
                    if os.path.exists(f):
                        zf.write(f, arcname=f"prediccion{ext}")
            return buffer.getvalue()

    st.download_button("Descargar GeoJSON", to_geojson_bytes(df_next), file_name="prediccion.geojson")
    st.download_button("Descargar Shapefile (.zip)", to_shapefile_bytes(df_next), file_name="prediccion.zip")
