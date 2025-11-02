# app.py — Aplikasi Streamlit Prediksi Kelulusan Mahasiswa
import streamlit as st
import joblib
import pandas as pd

# --- 1. LOAD MODEL & SCALER ---
try:
    # Memuat Model dan Scaler yang telah di-tuning dan bebas leakage
    model = joblib.load('model_kelulusan.pkl')
    scaler = joblib.load('scaler_kelulusan.pkl')
except FileNotFoundError:
    st.error("❌ File model tidak ditemukan. Pastikan 'model_kelulusan.pkl' dan 'scaler_kelulusan.pkl' ada di folder yang sama.")
    st.stop()

# --- 2. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")

st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.markdown("""
Aplikasi ini menggunakan **Random Forest Classifier (Tuned)** untuk memprediksi kelulusan 
berdasarkan data **nilai dan kehadiran mentah**.
""")

# --- 3. FORM INPUT DATA (8 FITUR) ---
st.header("🧾 Input Data Mahasiswa")

# Semua input HARUS sama dengan kolom di FEATURE_COLUMNS
attendance = st.slider("1️⃣ Kehadiran (%)", 0, 100, 90)
midterm = st.slider("2️⃣ Nilai UTS", 0, 100, 75)
final = st.slider("3️⃣ Nilai UAS", 0, 100, 80)
assign_avg = st.slider("4️⃣ Rata-rata Nilai Tugas", 0, 100, 85)
participation = st.slider("5️⃣ Partisipasi Kelas", 0, 100, 80)
study_hours = st.slider("6️⃣ Jam Belajar per Minggu", 0, 40, 10)
age = st.slider("7️⃣ Usia Mahasiswa", 17, 35, 20)
gender = st.selectbox("8️⃣ Jenis Kelamin", ["Laki-laki", "Perempuan"])

# Mapping gender ke numerik (0 atau 1)
gender_num = 0 if gender == "Laki-laki" else 1

# --- 4. KONVERSI INPUT KE DATAFRAME SESUAI TRAINING ---
input_data = pd.DataFrame([[
    attendance, midterm, final, assign_avg, participation,
    study_hours, age, gender_num
]], columns=[
    'attendance', 'midterm', 'final', 'assign_avg', 'participation',
    'study_hours', 'age', 'gender' # Urutan dan nama kolom HARUS SAMA dengan X
])

# --- 5. PREDIKSI ---
if st.button("🚀 Lakukan Prediksi", type="primary"):
    
    # Scaling input data menggunakan scaler yang SAMA saat training
    input_scaled = scaler.transform(input_data)
    
    # Prediksi probabilitas dan kelas
    pred_proba = model.predict_proba(input_scaled)[0]
    pred_class = model.predict(input_scaled)[0]

    st.subheader("📊 Hasil Prediksi")
    if pred_class == 1:
        st.success("✅ **Prediksi: LULUS TEPAT WAKTU!**")
        st.balloons()
    else:
        st.error("⚠️ **Prediksi: TIDAK LULUS TEPAT WAKTU**")

    st.markdown(f"**Probabilitas Lulus:** {pred_proba[1]*100:.2f}%")
    st.markdown(f"**Probabilitas Tidak Lulus:** {pred_proba[0]*100:.2f}%")

    st.divider()
    st.caption("Model: Random Forest Classifier (Tuned) — Deployment dengan Streamlit")