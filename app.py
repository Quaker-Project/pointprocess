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
ruta_zonas = st.file_uploader("Opcional: Subir shapefile de zonas con ID común (.zip)", type=["zip"])

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

# Carga covariables
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

# Carga shapefile zonas con ID común para join (opcional)
gdf_zonas = None
join_field = None
if ruta_zonas is not None:
    gdf_zonas = cargar_shapefile_zip(ruta_zonas)
    if gdf_zonas is not None:
        gdf_zonas = gdf_zonas.to_crs(gdf.crs)
        # El usuario debe especificar el campo ID común:
        posibles_campos = list(gdf_zonas.columns)
        join_field = st.selectbox("Selecciona el campo ID común para join con la rejilla", posibles_campos)
        if join_field not in gdf_zonas.columns:
            st.error("Campo seleccionado no existe en shapefile de zonas.")
            st.stop()
        # Asignar a cada celda el valor del campo join_field según la intersección espacial con zonas
        gdf_grid = gpd.sjoin(gdf_grid, gdf_zonas[[join_field, "geometry"]], how="left", predicate="intersects")
        gdf_grid = gdf_grid.rename(columns={join_field: join_field+"_zone"})
    else:
        st.warning("No se pudo cargar correctamente el shapefile de zonas.")

# Ahora, en lugar de hacer join espacial con covariables, hacemos merge por campo común si existe
if gdf_covars is not None and selected_vars and gdf_zonas is not None and join_field:
    # Asegurar que covariables tienen el mismo campo de zona y con mismo nombre que la rejilla
    if join_field not in gdf_covars.columns:
        st.error(f"El shapefile de covariables no tiene el campo {join_field} necesario para el join.")
        st.stop()
    gdf_covars = gdf_covars.rename(columns={join_field: join_field+"_zone"})

# Mensajes de info
st.write(f"Entrenando con meses desde {fecha_entreno_inicio} hasta {fecha_entreno_fin}")
st.write(f"Simulando mes: {mes_simulacion}")

if st.button("Ejecutar simulación"):

    data = []
    for m in tqdm(train_months, desc="Generando dataset de entrenamiento"):
        df_month = gdf[gdf["month"] == m]

        # Join espacial para etiquetar eventos en la rejilla
        joined = gpd.sjoin(gdf_grid, df_month, predicate='contains', how='left')
        joined["label"] = joined["index_right"].notnull().astype(int)
        grouped = joined.groupby("cell_id").agg(label=("label", "max")).reset_index()

        merged = pd.merge(grouped, gdf_grid, on="cell_id")
        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf.crs)

        # Agregar covariables mediante merge por campo común si existe
        if gdf_covars is not None and selected_vars:
            if join_field and join_field+"_zone" in merged.columns and join_field+"_zone" in gdf_covars.columns:
                merged = merged.merge(
                    gdf_covars[[join_field+"_zone"] + selected_vars].drop_duplicates(subset=join_field+"_zone"),
                    left_on=join_field+"_zone",
                    right_on=join_field+"_zone",
                    how="left"
                )
            else:
                st.warning("No hay campo común para unir covariables, se omite covariables en entrenamiento.")

        merged["month"] = str(m)
        data.append(merged)

    df_model = pd.concat(data, ignore_index=True)

    feature_vars = ["X", "Y"] + selected_vars if selected_vars else ["X", "Y"]
    X = df_model[feature_vars].fillna(0)
    y = df_model["label"]

    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X, y)

    # Preparar datos para simulación mes objetivo
    df_sim_mes = gdf_grid.copy()

    # Asignar zona a la rejilla si corresponde
    if gdf_zonas is not None and join_field:
        # ya hecho arriba, pero aseguramos aquí
        pass

    # Agregar covariables para predicción
    if gdf_covars is not None and selected_vars:
        if join_field and join_field+"_zone" in df_sim_mes.columns and join_field+"_zone" in gdf_covars.columns:
            df_sim_mes = df_sim_mes.merge(
                gdf_covars[[join_field+"_zone"] + selected_vars].drop_duplicates(subset=join_field+"_zone"),
                left_on=join_field+"_zone",
                right_on=join_field+"_zone",
                how="left"
            )
        else:
            st.warning("No hay campo común para unir covariables, predicción sin covariables.")

    # Variables predictoras para simulación
    X_pred = df_sim_mes[feature_vars].fillna(0)

    df_sim_mes["proba"] = model.predict_proba(X_pred)[:, 1]
    df_sim_mes["pred"] = (df_sim_mes["proba"] >= umbral).astype(int)

    # Evaluar si hay datos reales para ese mes
    df_real_mes = gdf[gdf["month"] == mes_sim]
    if not df_real_mes.empty:
        joined_real = gpd.sjoin(gdf_grid, df_real_mes, predicate="contains", how="left")
        joined_real["real"] = joined_real["index_right"].notnull().astype(int)
        eval_df = joined_real.groupby("cell_id").agg(real=("real", "max")).reset_index()
        eval_df = eval_df.merge(df_sim_mes[["cell_id", "pred"]], on="cell_id", how="left").fillna(0)

        precision = precision_score(eval_df["real"], eval_df["pred"])
        recall = recall_score(eval_df["real"], eval_df["pred"])
        f1 = f1_score(eval_df["real"], eval_df["pred"])

        st.write(f"Precisión: {precision:.3f}")
        st.write(f"Recall: {recall:.3f}")
        st.write(f"F1-score: {f1:.3f}")

        # Mapa
        fig, ax = plt.subplots(figsize=(12, 10))
        gdf_grid.boundary.plot(ax=ax, color="gray", linewidth=0.3)
        df_sim_mes.plot(column="proba", cmap="OrRd", ax=ax, legend=True, alpha=0.7)
        df_real_mes.plot(ax=ax, marker="x", color="blue", markersize=15, label="Delitos reales")
        plt.title(titulo_mapa)
        plt.legend()
        plt.axis('off')
        st.pyplot(fig)
    else:
        st.warning("No hay datos reales para evaluar el mes de simulación.")
