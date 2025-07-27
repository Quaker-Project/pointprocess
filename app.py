import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
import numpyro.distributions as dist
from bstpp.main import Hawkes_Model
import arviz as az
from scipy.spatial import cKDTree
import os

st.set_page_config(layout="wide")
st.title("📊 Análisis de Hurtos con Modelo Hawkes")

# ---------------------------------------
# 1. Subida de archivos
# ---------------------------------------
st.sidebar.header("📁 Cargar archivos")
uploaded_events = st.sidebar.file_uploader("HURTOS (Shapefile .shp)", type=["shp"])
uploaded_boundaries = st.sidebar.file_uploader("Red vial (Shapefile .shp)", type=["shp"])

if uploaded_events and uploaded_boundaries:
    with st.spinner("Leyendo archivos..."):
        gdf_events = gpd.read_file(uploaded_events).to_crs("EPSG:4326")
        gdf_boundaries = gpd.read_file(uploaded_boundaries).to_crs("EPSG:4326")
        gdf_events["Fecha"] = pd.to_datetime(gdf_events["Fecha"], format="%d/%m/%Y", errors="coerce")
        gdf_events["t"] = (gdf_events["Fecha"] - gdf_events["Fecha"].min()).dt.total_seconds() / 86400
        gdf_events = gdf_events.sort_values(by="t").reset_index(drop=True)

        # Dividir en train/test
        fecha_split = pd.to_datetime("2018-12-31")
        gdf_train = gdf_events[gdf_events["Fecha"] < fecha_split]
        gdf_test = gdf_events[gdf_events["Fecha"] >= fecha_split]

        # Buffer de red vial
        gdf_buffered = gdf_boundaries.copy()
        gdf_buffered["geometry"] = gdf_buffered.buffer(0.00015)

        st.success("✅ Datos cargados correctamente")

        # ---------------------------------------
        # 2. Entrenamiento
        # ---------------------------------------
        st.sidebar.header("⚙️ Entrenamiento del modelo")
        run_model = st.sidebar.button("Entrenar Modelo Hawkes")

        if run_model:
            with st.spinner("Entrenando modelo Hawkes..."):
                data_model = gdf_train[["t", "Long", "Lat"]].rename(columns={"t": "T", "Long": "X", "Lat": "Y"})
                model = Hawkes_Model(
                    data=data_model,
                    A=gdf_buffered,
                    T=gdf_train["t"].max(),
                    cox_background=True,
                    a_0=dist.Normal(1, 10),
                    alpha=dist.Beta(20, 60),
                    beta=dist.HalfNormal(2.0),
                    sigmax_2=dist.HalfNormal(0.25)
                )
                model.run_svi(lr=0.02, num_steps=2000)
                st.success("✅ Modelo entrenado correctamente")

                # Evaluación
                gdf_test = gdf_test.sort_values(by="t")
                data_test = gdf_test[["t", "Long", "Lat"]].rename(columns={"t": "T", "Long": "X", "Lat": "Y"})

                st.subheader("📈 Evaluación del Modelo")
                st.write("**Log Expected Likelihood:**", model.log_expected_likelihood(data_test))
                st.write("**Expected AIC:**", model.expected_AIC())

                # ---------------------------------------
                # 3. Visualizaciones
                # ---------------------------------------
                st.subheader("🗺️ Visualización Espacial del Background")
                fig1 = model.plot_spatial(include_cov=False)
                st.pyplot(fig1)

                st.markdown("Esta intensidad representa zonas con mayor probabilidad base de hurtos, independiente de eventos previos.")

                st.subheader("📈 Intensidad Temporal λ(t)")
                f_samples = model.samples["f_t"]
                times = np.linspace(0, model.T, f_samples.shape[1])
                lambda_samples = np.exp(f_samples)
                lambda_mean = lambda_samples.mean(axis=0)
                lambda_lower = np.percentile(lambda_samples, 5, axis=0)
                lambda_upper = np.percentile(lambda_samples, 95, axis=0)
                event_times = gdf_train["t"]

                fig2, ax = plt.subplots(figsize=(10, 5))
                ax.plot(times, lambda_mean, label="λ(t)", color="blue")
                ax.fill_between(times, lambda_lower, lambda_upper, color="blue", alpha=0.2)
                ax.plot(event_times, np.full_like(event_times, ax.get_ylim()[0]), "|", color="red", alpha=0.4)
                ax.set_title("Intensidad temporal λ(t)")
                ax.set_xlabel("Días desde el primer evento")
                ax.set_ylabel("Intensidad")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig2)

                st.subheader("📍 Dispersión Espacial del Trigger")
                fig3 = model.plot_trigger_posterior(trace=True)
                st.pyplot(fig3)

                st.subheader("⌛ Decaimiento Temporal del Trigger")
                fig4 = model.plot_trigger_time_decay()
                fig4.gca().set_ylim(0, 10)
                st.pyplot(fig4)

                st.subheader("🔥 Proporción de Eventos Autoexcitados")
                fig5 = model.plot_prop_excitation()
                st.pyplot(fig5)

                # ---------------------------------------
                # 4. Análisis Proxy (heurístico)
                # ---------------------------------------
                st.subheader("📌 Análisis Proxy de Autoexcitación")
                gdf_train["X"] = gdf_train.geometry.x
                gdf_train["Y"] = gdf_train.geometry.y

                coords = gdf_train[["X", "Y"]].to_numpy()
                times_array = gdf_train["t"].to_numpy()
                tree = cKDTree(coords)
                max_dist = 0.01
                max_time_diff = 7
                p_exc_proxy = np.zeros(len(gdf_train))

                for i, (xi, ti) in enumerate(zip(coords, times_array)):
                    idx = tree.query_ball_point(xi, max_dist)
                    idx = [j for j in idx if j != i and 0 < (ti - times_array[j]) <= max_time_diff]
                    p_exc_proxy[i] = 1.0 if idx else 0.0

                gdf_train["p_exc_proxy"] = p_exc_proxy

                fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                gdf_train.plot(ax=ax1, markersize=10, color="skyblue", alpha=0.5)
                gdf_train[gdf_train["p_exc_proxy"] == 1].plot(ax=ax2, markersize=10, color="red", alpha=0.7)
                ax1.set_title("🔵 Todos los eventos")
                ax2.set_title("🔴 Eventos autoexcitados (proxy)")
                st.pyplot(fig6)

                st.info("El análisis heurístico estima qué hurtos son inducidos por eventos cercanos en tiempo y espacio.")

else:
    st.warning("Por favor, sube los archivos `.shp` de eventos y red vial en la barra lateral.")
