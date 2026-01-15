# 🪙 Solana Price Prediction App

Aplikasi prediksi harga Solana menggunakan Regresi Semiparametrik (Nadaraya-Watson Kernel) dan Machine Learning (LightGBM).

## 📊 Project Overview

Project ini mengimplementasikan dan membandingkan dua metode prediksi time series:
1. **Nadaraya-Watson Kernel Regression** - Metode nonparametrik dengan kernel Gaussian
2. **LightGBM** - Gradient boosting machine learning untuk forecasting 1 jam ke depan

## 🎯 Fitur Utama

- ✅ **2 Metode Prediksi** - Perbandingan komprehensif NW Kernel vs LightGBM
- ✅ **Visualisasi Interaktif** - Dashboard lengkap dengan Plotly
- ✅ **Data Flexibility** - Upload CSV sendiri atau gunakan dataset default
- ✅ **Evaluasi Lengkap** - MSE, RMSE, MAPE, R², dan berbagai metrik
- ✅ **Feature Importance** - Analisis variabel paling berpengaruh
- ✅ **Responsive UI** - Design modern dan user-friendly

## 📁 Struktur Project

```
solana_prediction/
│
├── main.py                          # File utama untuk run app
│
├── data/
│   ├── df_1h.csv                    # Dataset (letakkan file Anda di sini)
│   └── data_loader.py               # Data loading & preprocessing
│
├── models/
│   ├── nadaraya_watson.py           # Implementasi NW Kernel
│   └── lightgbm_model.py            # Implementasi LightGBM
│
├── pages/
│   ├── 1_🏠_About.py                # Halaman About & Dataset
│   ├── 2_📚_Metodologi.py           # Penjelasan metodologi
│   ├── 3_📈_Dashboard.py            # Dashboard visualisasi
│   └── 4_👥_Developer.py            # Info tim developer
│
├── utils/
│   ├── visualizations.py            # Fungsi plotting
│   ├── metrics.py                   # Fungsi evaluasi model
│   └── styling.py                   # Custom CSS & styling
│
├── assets/
│   ├── team_info.py                 # Info anggota kelompok
│   └── dataset_info.py              # Info dataset & penelitian
│
├── requirements.txt                 # Python dependencies
└── README.md                        # File ini
```

## 🚀 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/solana-prediction.git
cd solana-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Letakkan file `df_1h.csv` Anda di folder `data/`:

```
data/
└── df_1h.csv
```

**Format CSV yang diperlukan:**
- Kolom: `datetime`, `open`, `high`, `low`, `close`, `volume`
- Datetime format: `YYYY-MM-DD HH:MM:SS`
- Semua nilai numeric untuk kolom harga dan volume

### 4. Run Application

```bash
streamlit run main.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📖 Cara Menggunakan

### Step 1: Load Data (Menu About)
- Pilih **"Gunakan Dataset Default"** atau **"Upload CSV Sendiri"**
- Lihat preview dan statistik dataset
- Eksplorasi visualisasi data

### Step 2: Pelajari Metodologi (Menu Metodologi)
- Baca penjelasan Nadaraya-Watson Kernel
- Pelajari cara kerja LightGBM
- Pahami ML process & preprocessing

### Step 3: Train & Visualisasi (Menu Dashboard)
- Klik tombol **"Train Models"**
- Tunggu hingga training selesai (~1-2 menit)
- Lihat hasil prediksi dan perbandingan
- Analisis visualisasi detail

### Step 4: Info Developer (Menu Developer)
- Lihat info tim dan project details

## 🔬 Metodologi

### Nadaraya-Watson Kernel Regression

**Formula:**
```
ŷ = Σ(K_h(x - x_i) * y_i) / Σ(K_h(x - x_i))
```

**Karakteristik:**
- Nonparametrik, tidak butuh asumsi distribusi
- Bandwidth optimal dipilih menggunakan GCV
- Kernel Gaussian untuk smooth estimation

### LightGBM Gradient Boosting

**Features:**
- Lag features (t-1) untuk prediksi 1 jam ke depan
- 500 estimators dengan learning rate 0.05
- Temporal split untuk menjaga urutan waktu

## 📊 Evaluasi Model

Aplikasi menghitung berbagai metrik:
- **Accuracy (%)** - 100 - MAPE
- **MAPE (%)** - Mean Absolute Percentage Error
- **RMSE** - Root Mean Squared Error
- **R² Score** - Coefficient of Determination
- **MAE** - Mean Absolute Error

## 🛠️ Technology Stack

- **Frontend:** Streamlit
- **Visualization:** Plotly, Matplotlib
- **ML Framework:** LightGBM, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Language:** Python 3.8+

## 👥 Team

Project ini dikembangkan oleh kelompok Regresi Semiparametrik:

1. **Anggota 1** - Project Manager & Data Analyst
2. **Anggota 2** - Machine Learning Engineer
3. **Anggota 3** - Statistical Modeling Specialist
4. **Anggota 4** - Full Stack Developer & UI/UX
5. **Anggota 5** - Data Visualization & Report Writer

*Edit file `assets/team_info.py` untuk update info tim*

## 📝 Customization

### Update Info Tim
Edit file `assets/team_info.py`:
```python
TEAM_MEMBERS = [
    {
        "nama": "Nama Anda",
        "nim": "NIM Anda",
        "role": "Role Anda",
        "email": "email@domain.com",
        "responsibilities": [...]
    },
    ...
]
```

### Update Project Info
Edit file `assets/team_info.py`:
```python
PROJECT_INFO = {
    "course_name": "Nama Mata Kuliah",
    "university": "Nama Universitas",
    "lecturer": "Nama Dosen",
    ...
}
```

### Ganti Dataset Default
Replace file `data/df_1h.csv` dengan dataset Anda.

## 🐛 Troubleshooting

### Error: Module not found
```bash
pip install -r requirements.txt
```

### Error: File not found (df_1h.csv)
- Pastikan file ada di folder `data/`
- Atau gunakan fitur upload CSV

### Error: Memory error saat training
- Kurangi ukuran dataset
- Atau kurangi `n_estimators` di LightGBM

### Aplikasi lambat
- Kurangi jumlah kandidat bandwidth untuk NW
- Gunakan dataset yang lebih kecil untuk testing

## 📚 References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Nadaraya-Watson Kernel Regression](https://en.wikipedia.org/wiki/Kernel_regression)

## 📄 License

Project ini dibuat untuk keperluan akademik - Mata Kuliah Regresi Semiparametrik.

## 🙏 Acknowledgments

- Dosen pengampu: [Nama Dosen]
- Data provider: Cryptocurrency Exchange API
- Open-source community

---

**Made with ❤️ by Kelompok Regresi Semiparametrik 2024/2025**