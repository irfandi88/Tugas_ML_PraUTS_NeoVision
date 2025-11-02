# =================================================================
# TRAINING MODEL MACHINE LEARNING (Random Forest)
# Topik: Prediksi Berdasarkan NILAI dan KEHADIRAN 
# =================================================================

# --- 1. IMPORT LIBRARIES ---
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Abaikan warning minor
warnings.filterwarnings('ignore')

# --- 2. LOAD DATASET & DATA UNDERSTANDING ---
try:
    df = pd.read_csv('Tugas_ML_PraUTS_NeoVision/dataset.csv') 
    print(" Dataset 'dataset.csv' berhasil dimuat.")
except FileNotFoundError:
    print(" Error: File 'dataset.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    exit()

print(f"Jumlah data: {df.shape[0]} baris.")


# --- 3. DATA PREPROCESSING (FOKUS KE NILAI MENTAH) ---
X = df.drop(['lulus', 'weighted_score'], axis=1)
y = df['lulus']

# Simpan urutan dan nama kolom fitur yang digunakan untuk memastikan konsistensi saat deployment
FEATURE_COLUMNS = list(X.columns)

# Split data (80% train, 20% test)
# stratify=y memastikan rasio Lulus/Tidak Lulus seimbang di set train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\n Data berhasil dibagi: 80% Training, 20% Testing (Stratified).")

# Scaling Fitur Numerik
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(" Data berhasil di-scaling (StandardScaler).")


# --- 4. MODEL TRAINING DENGAN HYPERPARAMETER TUNING (KOMPLEKSITAS TINGGI) ---

# Tentukan parameter yang akan diuji
param_grid = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [5, 8, 12],         
    'min_samples_leaf': [1, 5],      
}

# Gunakan GridSearchCV (Grid Search Cross-Validation)
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=5, # 5-Fold Cross-Validation: Evaluasi lebih stabil
    scoring='accuracy'
)

print("\n Melakukan Hyperparameter Tuning (Grid Search 5-Fold)...")
grid_search.fit(X_train_scaled, y_train)

# Ambil model terbaik
rf_model_final = grid_search.best_estimator_
print(f" Tuning Selesai. Parameter Terbaik: {grid_search.best_params_}")


# --- 5. MODEL EVALUATION (Menggunakan Model Terbaik) ---
y_pred = rf_model_final.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("\n===================================")
print(" HASIL EVALUASI MODEL FINAL")
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


# --- 6. ANALISIS FEATURE IMPORTANCE (FOKUS KE NILAI DAN KEHADIRAN) ---

importances = rf_model_final.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': FEATURE_COLUMNS,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n===================================")
print(" ANALISIS FEATURE IMPORTANCE")
print("===================================")
print("Analisis ini menunjukkan fitur nilai/kehadiran yang paling penting:")
print(feature_importance_df)

# Visualisasi dan Simpan Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
plt.title('Kepentingan Fitur dalam Memprediksi Kelulusan (Random Forest Terbaik)')
plt.xlabel('Tingkat Kepentingan')
plt.ylabel('Fitur')
plt.tight_layout()
plt.savefig('feature_importance_plot_FINAL.png')
print("\n Plot Feature Importance berhasil disimpan sebagai 'feature_importance_plot_FINAL.png'.")

# --- 7. SIMPAN ARTIFACTS (MODEL & SCALER) ---
joblib.dump(rf_model_final, 'model_kelulusan.pkl') 
joblib.dump(scaler, 'scaler_kelulusan.pkl')

print("\n===================================")
print(" PENYIMPANAN ARTIFACTS")
print("===================================")
print(" Model terbaik berhasil disimpan: 'model_kelulusan.pkl'")
print(" Scaler berhasil disimpan: 'scaler_kelulusan.pkl'")
print(f"Fitur yang digunakan untuk Deployment: {FEATURE_COLUMNS}")