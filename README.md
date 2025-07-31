# Income Classification with Machine Learning & Flask Deployment

Proyek ini memuat dua bagian utama:

1. **Data Mining & Machine Learning** — Meliputi proses KDD : pembersihan data, rekayasa fitur, training model klasifikasi pendapatan, hingga evaluasi performa.
2. **Flask Web Deployment** — Aplikasi web sederhana untuk menerima input pengguna dan melakukan prediksi pendapatan berdasarkan model yang telah dilatih (deployment). 

## Bagian 1: Data Mining & Model Training

Semua eksperimen machine learning dilakukan menggunakan Jupyter Notebook.

### Tahapan Proses

- **Exploratory Data Analysis (EDA)**
- **Data Preparation:**
  - Menangani missing values
  - Menghapus data duplikat
  - Rekayasa fitur (feature engineering)
  - Encoding untuk data kategorikal
  - Menghapus fitur yang tidak relevan
  - Penanganan outlier dengan teknik capping
  - Feature scaling menggunakan StandardScaler
- **Splitting Data** menjadi Train/Test
- **Hybrid Resampling**:
  - Undersampling untuk kelas mayoritas
  - SMOTE (Synthetic Minority Oversampling Technique) untuk kelas minoritas
- **Model Training dan Evaluation**
  - Menggunakan model klasifikasi seperti Random Forest, Logistic Regression, dll.
  - Evaluasi performa model menggunakan metrik akurasi, F1-score, recall, dll.
- **Model Saving**
  - Model disimpan dalam format `.pkl`

### File Penting

- `adult.csv` – Dataset mentah
- `FinalCodeDM.ipynb` – Notebook utama preprocessing dan pelatihan model

---

## Bagian 2: Web Deployment dengan Flask

Aplikasi Flask menyediakan antarmuka pengguna untuk mengisi data input dan melihat hasil prediksi pendapatan.

### ⚙️ Cara Menjalankan Aplikasi Flask

1. **Aktifkan virtual environment (venv) dan jalankan web:**
   venv\Scripts\activate
   pip install -r requirements.txt
   python app.py
2. **Buka dan Akses Port**




