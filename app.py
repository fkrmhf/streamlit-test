import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ======================================
# LOAD MODEL
# ======================================
@st.cache_resource
def load_models():
    linear_model = joblib.load("linear_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    return linear_model, rf_model

# Try load model
try:
    linear_model, rf_model = load_models()
    model_ready = True
except:
    model_ready = False


# ======================================
# STREAMLIT UI
# ======================================
st.set_page_config(page_title="SmartHome Energy AI", layout="wide")

st.title("⚡ SmartHome Energy Prediction AI")
st.write("Aplikasi prediksi konsumsi energi harian untuk integrasi SmartHome.")

st.markdown("---")


# ======================================
# INPUT SECTION
# ======================================
st.header("📥 Input Data Harian")

st.write("""
Masukkan fitur-fitur agregat harian.  
Biasanya data ini dihasilkan dari sensor SmartHome (AC, Heater, Laundry, Voltage, dsb.)
""")

# Sidebar inputs
gap_mean = st.number_input("Global Active Power (mean, kW)", 0.0, 10.0, 1.2)
grp_mean = st.number_input("Global Reactive Power (mean, kW)", 0.0, 5.0, 0.1)
voltage_mean = st.number_input("Voltage (mean)", 200.0, 300.0, 240.0)
current_mean = st.number_input("Current (mean, A)", 0.0, 40.0, 5.0)
sub1_sum = st.number_input("Sub Metering 1 (Wh)", 0.0, 30000.0, 1200.0)
sub2_sum = st.number_input("Sub Metering 2 (Wh)", 0.0, 30000.0, 800.0)
sub3_sum = st.number_input("Sub Metering 3 (Wh)", 0.0, 30000.0, 3000.0)
dayofweek = st.selectbox("Day of Week", [0,1,2,3,4,5,6])
month = st.selectbox("Month", list(range(1,13)))

# Lag inputs
st.subheader("⏱️ Lag Features (Energi 7 Hari Sebelumnya, kWh)")
lag_features = []
for i in range(1,8):
    lag = st.number_input(f"Lag {i} (kWh)", 0.0, 50.0, float(5+i))
    lag_features.append(lag)


# ======================================
# PREDICTION
# ======================================
if st.button("🔮 Prediksi Konsumsi Energi"):
    if not model_ready:
        st.error("Model belum tersedia. Upload linear_model.pkl dan rf_model.pkl")
    else:
        input_data = pd.DataFrame([[
            gap_mean, grp_mean, voltage_mean, current_mean,
            sub1_sum, sub2_sum, sub3_sum,
            dayofweek, month,
            *lag_features
        ]], columns=[
            "gap_mean", "grp_mean", "voltage_mean", "current_mean",
            "sub1_sum", "sub2_sum", "sub3_sum",
            "dayofweek", "month",
            "lag_1","lag_2","lag_3","lag_4","lag_5","lag_6","lag_7"
        ])

        # Predict
        pred_lin = linear_model.predict(input_data)[0]
        pred_rf = rf_model.predict(input_data)[0]

        st.success(f"Prediksi Linear Regression: **{pred_lin:.2f} kWh**")
        st.success(f"Prediksi Random Forest: **{pred_rf:.2f} kWh**")

        st.markdown("---")

        # ======================================
        # INSIGHTS FOR USER
        # ======================================
        st.header("💡 Insight SmartHome dari AI")

        insight = []

        # Example insights
        if pred_rf > max(lag_features):
            insight.append("Konsumsi energi hari ini diprediksi lebih tinggi dari rata-rata 7 hari terakhir.")
        else:
            insight.append("Konsumsi energi hari ini diprediksi stabil atau lebih rendah dari rata-rata minggu ini.")

        if sub3_sum > 2500:
            insight.append("Water Heater / AC kemungkinan menjadi sumber energi terbesar hari ini.")

        if current_mean > 10:
            insight.append("Arus listrik cukup tinggi, periksa penggunaan alat berat.")

        if month in [6,7,8]:
            insight.append("Musim panas → konsumsi AC biasanya meningkat.")

        if dayofweek in [5,6]:
            insight.append("Weekend biasanya lebih boros energi, pertimbangkan mode hemat.")

        st.write("### Rekomendasi AI:")
        for i in insight:
            st.write(f"- {i}")

        st.markdown("---")

        # ======================================
        # SIMPLE VISUALIZATION
        # ======================================
        st.header("📈 Visualisasi")

        # Combine into series
        past = lag_features[::-1] 
        future = [pred_rf]

        plt.figure(figsize=(10,4))
        plt.plot(range(-7,0), past, label="7 Hari Terakhir")
        plt.plot([0], future, "ro", label="Prediksi Hari Ini")
        plt.title("Prediksi Konsumsi Energi")
        plt.xlabel("Hari (negatif = masa lalu)")
        plt.ylabel("kWh")
        plt.legend()
        st.pyplot(plt)
