# app.py — Aplikasi Streamlit Prediksi Kelulusan Mahasiswa
import streamlit as st # type: ignore
import joblib
import pandas as pd

# --- 1. SETUP ARTIFACTS & KONFIGURASI ---

# DAFTAR FITUR HARUS SAMA PERSIS DENGAN URUTAN SAAT TRAINING
FEATURE_COLUMNS = [
    'attendance', 'midterm', 'final', 'assign_avg', 'participation',
    'study_hours', 'age', 'gender', 'weighted_score'
]

try:
    # Load Model dan Scaler
    model = joblib.load('model_kelulusan.pkl')
    scaler = joblib.load('scaler_kelulusan.pkl')
except FileNotFoundError:
    st.error("❌ Error: File model atau scaler tidak ditemukan. Pastikan 'model_kelulusan.pkl' dan 'scaler_kelulusan.pkl' ada di folder yang sama.")
    st.stop()

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")

st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.markdown("""
Aplikasi ini menggunakan **Random Forest Classifier** untuk memprediksi apakah mahasiswa **akan lulus tepat waktu atau tidak**.
""")

# --- 2. FORM INPUT DATA (Menggunakan st.form untuk efisiensi) ---
st.header("🧾 Input Data Mahasiswa")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Kategori Nilai")
        midterm = st.slider("Nilai UTS (0-100)", 0, 100, 75)
        final = st.slider("Nilai UAS (0-100)", 0, 100, 80)
        assign_avg = st.slider("Rata-rata Nilai Tugas (0-100)", 0, 100, 85)
        weighted_score = st.slider("Weighted Score (Gabungan nilai) (0-100)", 0, 100, 85)

    with col2:
        st.markdown("#### Kategori Non-Nilai")
        attendance = st.slider("Kehadiran (%) (0-100)", 0, 100, 90)
        participation = st.slider("Partisipasi Kelas (0-100)", 0, 100, 80)
        study_hours = st.slider("Jam Belajar per Minggu (0-40)", 0, 40, 10)
        age = st.slider("Usia Mahasiswa (17-35)", 17, 35, 20)
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki (0)", "Perempuan (1)"])

    # Mapping gender (mengambil angka 0 atau 1 dari string)
    gender_num = 1 if "Perempuan" in gender else 0

    # Tombol submit form
    submitted = st.form_submit_button("🚀 Lakukan Prediksi", type="primary")

# --- 3. PREDIKSI LOGIC ---
if submitted:
    # Kumpulkan input values sesuai urutan FEATURE_COLUMNS
    input_values = [
        attendance, midterm, final, assign_avg, participation,
        study_hours, age, gender_num, weighted_score
    ]

    # Buat DataFrame dengan kolom yang sesuai
    input_data = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)

    # Transformasi data dengan scaler yang sudah dilatih
    input_scaled = scaler.transform(input_data)

    # Prediksi
    pred_proba = model.predict_proba(input_scaled)[0]
    pred_class = model.predict(input_scaled)[0]

    st.subheader("📊 Hasil Prediksi")

    if pred_class == 1:
        st.success("✅ **Prediksi: LULUS TEPAT WAKTU!**")
        st.balloons()
    else:
        st.error("⚠️ **Prediksi: TIDAK LULUS TEPAT WAKTU**")

    # Tampilkan Probabilitas
    st.markdown(f"**Probabilitas Lulus (1):** `{pred_proba[1]*100:.2f}%`")
    st.markdown(f"**Probabilitas Tidak Lulus (0):** `{pred_proba[0]*100:.2f}%`")

    st.divider()
    st.caption("Metode: Random Forest Classifier — Deployment: Streamlit")