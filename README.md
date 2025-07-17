# streamlit_demandforecasting_sarimax_nn
A Streamlit app for batch demand forecasting
# Hybrid Forecasting SARIMAX + Neural Network (Streamlit)

This Streamlit app predicts product sales (currently WAFER) using a hybrid time series model combining SARIMAX and a Neural Network trained on residuals.

## Files
- `streamlit_app.py` — Main Streamlit app script
- `hybrid_wafers_components.pkl` — Trained hybrid model for WAFER
- `exog_wafers2.csv` — Sample exogenous data input for forecasting
- `requirements.txt` — Dependencies for deployment

## How to run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
