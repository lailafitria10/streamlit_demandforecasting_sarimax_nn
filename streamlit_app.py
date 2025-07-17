#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =====================================
# Hybrid Model Class (with adjusted prediction range)
# =====================================
class HybridSARIMAXNN:
    def __init__(self, sarimax_model, nn_model, scaler):
        self.sarimax = sarimax_model
        self.nn = nn_model
        self.scaler = scaler

    def predict(self, X_sarimax, X_nn):
        # Mulai prediksi SARIMAX setelah data training berakhir
        start = len(self.sarimax.data.endog)
        end = start + len(X_sarimax) - 1

        sarimax_pred = self.sarimax.predict(start=start, end=end, exog=X_sarimax)

        # Neural Net prediction
        X_nn_scaled = self.scaler.transform(X_nn)
        nn_pred = self.nn.predict(X_nn_scaled).flatten()

        return sarimax_pred + nn_pred

# =====================================
# Streamlit App
# =====================================
st.title("🍘 Hybrid Forecasting: SARIMAX + Neural Network untuk Produk WAFER")
st.write("Prediksi penjualan produk *wafer* menggunakan model hybrid SARIMAX + Neural Network (NN).")

# Load Model Bundle
@st.cache_resource
def load_model_bundle():
    with open("hybrid_wafers_components.pkl", "rb") as f:
        bundle = pickle.load(f)
    nn_model = load_model("nn_wafers_model.h5")
    hybrid_model = HybridSARIMAXNN(bundle["sarimax"], nn_model, bundle["scaler"])
    return hybrid_model

model = load_model_bundle()

# Upload file fitur exogenous
uploaded_file = st.file_uploader("📁 Upload file exogenous (exog_wafers2.csv)", type="csv")

if uploaded_file is not None:
    df_exog = pd.read_csv(uploaded_file)

    st.write("📊 Data Exogenous yang Diupload:")
    st.dataframe(df_exog.head())

    # Tombol Prediksi
    if st.button("🔮 Prediksi Penjualan Wafer"):
        try:
            X_sarimax = df_exog.copy()
            X_nn = df_exog.copy()

            # Prediksi hybrid
            forecast = model.predict(X_sarimax, X_nn)

            # Tampilkan hasil prediksi
            st.subheader("📈 Hasil Prediksi Penjualan Wafer")
            st.line_chart(forecast)

            # Bandingkan dengan aktual jika tersedia
            if 'actual' in df_exog.columns:
                st.subheader("📊 Perbandingan Forecast vs Aktual")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_exog['actual'].values, label="Actual", linewidth=2)
                ax.plot(forecast, label="Forecast", linestyle="--")
                ax.legend()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
