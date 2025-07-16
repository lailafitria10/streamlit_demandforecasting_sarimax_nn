#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Wrapper class untuk Hybrid Model
class HybridSARIMAXNN:
    def __init__(self, sarimax_model, nn_model, scaler):
        self.sarimax = sarimax_model
        self.nn = nn_model
        self.scaler = scaler

    def predict(self, X_sarimax, X_nn):
        sarimax_pred = self.sarimax.predict(start=0, end=len(X_sarimax)-1, exog=X_sarimax)
        X_nn_scaled = self.scaler.transform(X_nn)
        nn_pred = self.nn.predict(X_nn_scaled).flatten()
        return sarimax_pred.values + nn_pred

# Judul halaman
st.title("🍘 Hybrid Forecasting: SARIMAX + Neural Network untuk Produk WAFER")
st.write("Prediksi penjualan produk *wafer* menggunakan model hybrid SARIMAX + Neural Network (NN).")

# Load 1 model hybrid (wafer)
@st.cache_resource
def load_model():
    with open("hybrid_model_wafer.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Upload file fitur exogenous
uploaded_file = st.file_uploader("📁 Upload file exogenous (exog_wafers.csv)", type="csv")

if uploaded_file is not None:
    df_exog = pd.read_csv(uploaded_file)

    st.write("📊 Data Exogenous yang Diupload:")
    st.dataframe(df_exog.head())

    # Tombol prediksi
    if st.button("🔮 Prediksi Penjualan Wafer"):
        try:
            X_sarimax = df_exog.copy()
            X_nn = df_exog.copy()

            # Prediksi hybrid
            forecast = model.predict(X_sarimax, X_nn)

            # Tampilkan hasil
            st.subheader("📈 Hasil Prediksi Penjualan Wafer")
            st.line_chart(forecast)

            # Bandingkan dengan aktual jika ada
            if 'actual' in df_exog.columns:
                st.subheader("📊 Perbandingan Forecast vs Aktual")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_exog['actual'].values, label="Actual", linewidth=2)
                ax.plot(forecast, label="Forecast", linestyle="--")
                ax.legend()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")


# In[ ]:




