# 🎓 AI-Powered Exam Cheating Detection (MLC Data Slayer 4.0)

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Validation](https://img.shields.io/badge/Validation-Stratified%205--Fold-orange.svg)](https://scikit-learn.org/)
[![SDGs](https://img.shields.io/badge/SDGs-4%20Quality%20Education-orange.svg)](https://sdgs.un.org/goals/goal4)

Repository ini berisi implementasi solusi **Machine Learning** untuk mendeteksi indikasi kecurangan dalam sistem ujian online (*Computer-Based Test / CBT*), disusun berdasarkan tema kompetisi **Data Slayer 4.0: *"Honest Score, Equal Chance: Exam Cheating Detection"*** (Himpunan Mahasiswa Sains Data - Telkom University Purwokerto).

---

## 📌 Latar Belakang & Masalah
Ujian online rentan terhadap aksi kecurangan terorganisir, mulai dari penyalinan jawaban (*copy-paste*), membuka jendela browser lain (*tab switching*), hingga penggunaan joki melalui koneksi remote (*IP switching*). Proyek ini merancang sistem deteksi anomali berbasis Machine Learning yang objektif, transparan (*Explainable AI*), dan memiliki performa tinggi tanpa perlu memanggil API pihak ketiga (100% *self-contained / local*).

---

## 🔬 Hipotesis Domain & Rekayasa Fitur (Feature Engineering)
Dalam proyek ini, dilakukan pengujian bertahap terhadap 2 hipotesis domain pendidikan:

1. **Ide 1: Speedrun Cheating Anomaly**  
   * **Hipotesis:** Siswa yang mendapatkan nilai tinggi ($\ge 90$) dalam durasi sangat singkat ($\le 25$ menit dari alokasi 90 menit) menunjukkan indikasi kuat kecurangan.
   * **Rumus:** `score_speed_ratio = score / duration_minutes` dan flag biner `is_suspicious_speed`.
2. **Ide 2: Joki & Remote Access Anomaly**  
   * **Hipotesis:** Siswa yang mengalami pergantian alamat IP internet di tengah sesi ujian dan secara bersamaan sering berpindah jendela tab browser memiliki probabilitas sangat tinggi memakai jasa joki remote.
   * **Rumus:** `joki_remote_anomaly = network_ip_changes * tab_switch_count` dan flag biner `is_suspicious_joki`.

---

## 🏆 Hasil Turnamen Eksperimen (Stratified 5-Fold CV)

Seluruh model diuji menggunakan **5-Fold Stratified Cross-Validation** dengan seed terkunci (`seed=42`) sesuai regulasi lomba (Halaman 6 Butir 9).

| Model | Fitur yang Digunakan | ROC-AUC | F1-Score | Keterangan |
| :--- | :--- | :---: | :---: | :--- |
| **Model A** | Data Mentah Panitia | `0.9930` | `85.79%` | Baseline awal |
| **Model B** | Baseline + **Ide 1 (Speedrun Cheat)** | `0.9929` | `85.52%` | Fokus pada waktu & nilai |
| **Model C** | Baseline + **Ide 1 & Ide 2 (Joki Remote)** | **`0.9941`** | **`87.74%`** | **Model Terbaik (+1.95% F1-Score)** 🚀 |

---

## 📂 Struktur Proyek
```text
mlc-cheating-detection/
├── data/
│   ├── train.csv                # 3.500 data log latih dengan label
│   ├── test.csv                 # 1.500 data log uji tanpa label
│   └── sample_submission.csv    # Format acuan submisi
├── notebooks/
│   └── 01_cheating_detection_solution.ipynb  # Notebook eksperimen & visualisasi lengkap
├── output/
│   ├── final_exam_cheating_model.pkl         # Model ensemble tersimpan (.pkl)
│   └── submission_final.csv                  # File prediksi final untuk Kaggle Leaderboard
├── generate_data.py             # Script generator dataset simulasi
├── train_baseline.py            # Script pipeline pelatihan end-to-end
├── FINAL_PRESENTATION_DECK.md   # Panduan slide presentasi babak final (10 menit)
├── TECHNICAL_REPORT.md          # Draf laporan teknis (Technical Report) babak final
├── .gitignore                   # Konfigurasi file terabaikan Git
└── README.md                    # Dokumentasi portofolio profesional
```

---

## 🚀 Panduan Menjalankan Proyek
1. Clone repositori:
   ```bash
   git clone https://github.com/ayubwicak/mlc-cheating-detection.git
   cd mlc-cheating-detection
   ```
2. Buat virtual environment dan pasang pustaka:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install numpy pandas scikit-learn lightgbm matplotlib seaborn joblib ipykernel
   ```
3. Jalankan notebook interaktif di VS Code / Jupyter:
   ```bash
   code notebooks/01_cheating_detection_solution.ipynb
   ```
