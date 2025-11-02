# =======================================================
# SCRIPT TRAINING MODEL MACHINE LEARNING (Random Forest)
# Prediksi Kelulusan Mahasiswa Berdasarkan Nilai & Kehadiran
# =======================================================

# --- 1. IMPORT LIBRARIES ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Abaikan warning (misalnya terkait joblib)
warnings.filterwarnings('ignore')

# --- 2. LOAD DATASET & DATA UNDERSTANDING ---
try:
    df = pd.read_csv('Tugas_ML_PraUTS_NeoVision/dataset.csv')
    print(" Dataset 'dataset.csv' berhasil dimuat.")
except FileNotFoundError:
    print(" Error: File 'dataset.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    exit()

print(f"Jumlah data: {df.shape[0]} baris.")
# Cek Missing Values (Jika ada)
if df.isnull().sum().sum() > 0:
    print("\n Peringatan: Ditemukan Missing Values. Silakan tangani terlebih dahulu.")
else:
    print(" Tidak ditemukan Missing Values.")


# --- 3. DATA PREPROCESSING ---

# Tentukan Fitur (X) dan Target (y)
X = df.drop('lulus', axis=1)
y = df['lulus']

# Simpan urutan kolom fitur yang digunakan untuk memastikan konsistensi saat deployment
FEATURE_COLUMNS = list(X.columns)

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n Data berhasil dibagi: 80% Training, 20% Testing.")

# Scaling Fitur Numerik
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(" Data berhasil di-scaling (StandardScaler).")


# --- 4. MODEL TRAINING (Random Forest Classifier) ---

rf_model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
print("\n Model Random Forest sedang dilatih...")
rf_model.fit(X_train_scaled, y_train)
print(" Model berhasil dilatih.")


# --- 5. MODEL EVALUATION ---
y_pred = rf_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("\n===================================")
print("      HASIL EVALUASI MODEL")
print("===================================")
print(f" Akurasi Model (Test Set): {accuracy*100:.2f}%")

# Cek Kriteria (> 90%)
if accuracy > 0.90:
    print(" STATUS: Akurasi MEMENUHI kriteria (> 90%).")
else:
    print(" STATUS: Akurasi BELUM mencapai kriteria (> 90%).")

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))


# --- 6. ANALISIS FEATURE IMPORTANCE ---

importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': FEATURE_COLUMNS,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n===================================")
print("      ANALISIS FEATURE IMPORTANCE")
print("===================================")
print(feature_importance_df)

# Visualisasi dan Simpan Plot (Wajib untuk Presentasi)
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df,
    palette='magma'
)
plt.title('Kepentingan Fitur dalam Memprediksi Kelulusan (Random Forest)')
plt.xlabel('Tingkat Kepentingan')
plt.ylabel('Fitur')
plt.tight_layout()
plt.savefig('feature_importance_plot.png')
print("\n Plot Feature Importance berhasil disimpan sebagai 'feature_importance_plot.png'.")
# plt.show() # Hapus jika hanya dijalankan di server tanpa GUI

# --- 7. SIMPAN ARTIFACTS (MODEL & SCALER) ---

joblib.dump(rf_model, 'model_kelulusan.pkl')
joblib.dump(scaler, 'scaler_kelulusan.pkl')

print("\n===================================")
print("      PENYIMPANAN ARTIFACTS")
print("===================================")
print(" Model berhasil disimpan: 'model_kelulusan.pkl'")
print(" Scaler berhasil disimpan: 'scaler_kelulusan.pkl'")
print(f"Fitur yang digunakan: {FEATURE_COLUMNS}")