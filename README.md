# 🎓 AI-Powered Online Exam Cheating Detection (CBT Anomaly Detection)

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Validation](https://img.shields.io/badge/Validation-Stratified%205--Fold-orange.svg)](https://scikit-learn.org/)
[![SDGs](https://img.shields.io/badge/SDGs-4%20Quality%20Education-orange.svg)](https://sdgs.un.org/goals/goal4)

Repository ini berisi implementasi solusi **Machine Learning** end-to-end untuk mendeteksi indikasi kecurangan dan anomali perilaku dalam sistem ujian online (*Computer-Based Test / CBT*). Proyek ini dirancang sebagai studi kasus komprehensif untuk mendemonstrasikan keahlian dalam *Exploratory Data Analysis (EDA)*, *Hypothesis Testing*, *Feature Engineering*, *Stratified Cross-Validation*, dan *Explainable AI (XAI)*.

---

## 📌 Latar Belakang & Masalah
Peralihan menuju ujian berbasis komputer membawa kemudahan evaluasi massal, namun memunculkan celah kecurangan:
1. Membuka tab browser lain untuk mencari kunci jawaban (*tab switching*).
2. Menyalin soal atau menempelkan jawaban dari sumber eksternal (*copy-paste*).
3. Menggunakan bantuan joki remote melalui pergantian jaringan (*dynamic IP / remote access*).
4. Menyelesaikan ujian berkecepatan ekstrem dengan nilai sempurna (*speedrun cheat*).

Proyek ini membangun model deteksi berbasis telemetri aktivitas yang **objektif, cepat, transparan, dan tidak bergantung pada API pihak ketiga**.

---

## 🔬 Formulasi Hipotesis & Rekayasa Fitur (Feature Engineering)
Dalam proyek ini, dilakukan pengujian bertahap terhadap hipotesis domain pendidikan:

1. **Ide 1: Speedrun Cheating Anomaly**  
   * **Hipotesis:** Mahasiswa yang meraih skor tinggi ($\ge 90$) dalam durasi sangat singkat ($\le 25$ menit dari batas 90 menit) menunjukkan perilaku abnormal.
   * **Implementasi:** `score_speed_ratio = score / duration_minutes` dan flag biner `is_suspicious_speed`.
2. **Ide 2: Joki & Remote Access Anomaly**  
   * **Hipotesis:** Mahasiswa yang mengalami pergantian alamat IP internet di tengah sesi ujian dan secara simultan sering berganti jendela tab browser memiliki probabilitas sangat tinggi memakai jasa joki/remote desktop.
   * **Implementasi:** `joki_remote_anomaly = network_ip_changes * tab_switch_count` dan flag biner `is_suspicious_joki`.

---

## 🏆 Hasil Turnamen Eksperimen (Stratified 5-Fold CV)

Seluruh model dievaluasi secara ketat menggunakan **5-Fold Stratified Cross-Validation** dengan seed terkunci (`seed=42`) untuk menjamin konsistensi reproduksi (*reproducibility*):

| Model | Fitur yang Digunakan | ROC-AUC | F1-Score | Keterangan |
| :--- | :--- | :---: | :---: | :--- |
| **Model A** | Data Mentah Dasar | `0.9930` | `85.79%` | Baseline awal |
| **Model B** | Baseline + **Ide 1 (Speedrun Cheat)** | `0.9929` | `85.52%` | Fokus pada rasio waktu & nilai |
| **Model C** | Baseline + **Ide 1 & Ide 2 (Joki Remote)** | **`0.9941`** | **`87.74%`** | **Model Terbaik (+1.95% F1-Score)** 🚀 |

> **Catatan metodologis:** F1 pada tabel di atas dihitung pada threshold
> default `0.5`. Pada repeated CV dan threshold tuning, Logistic
> Regression dengan fitur mentah justru unggul di PR-AUC/F1, sedangkan
> LightGBM Model C unggul di ROC-AUC. Artinya, pemilihan "model terbaik"
> sangat bergantung pada metrik dan threshold. Detail eksperimen ada di
> [`MODELING_GUIDE.md`](MODELING_GUIDE.md).

---

## 📂 Struktur Proyek
```text
exam-cheating-detection-ml/
├── data/
│   ├── train.csv                # Data log latih (3.500 sampel dengan label)
│   ├── test.csv                 # Data log uji (1.500 sampel tanpa label)
│   └── sample_submission.csv    # Format acuan prediksi
├── notebooks/
│   ├── 01_cheating_detection_solution.ipynb  # Notebook eksplorasi & eksperimen interaktif
│   └── 02_portfolio_overview.ipynb           # Ringkasan proyek untuk portofolio
├── src/
│   ├── config.py                # Path, seed, dan konstanta global
│   ├── features.py              # Feature engineering + definisi feature set A/B/C
│   ├── models.py                # Registry model (tambah model via decorator)
│   ├── evaluation.py            # Repeated CV, threshold tuning, metrik, bootstrap
│   ├── train.py                 # Turnamen model via CLI
│   └── predict.py               # Inference dari bundle model
├── output/
│   ├── final_exam_cheating_model.pkl         # Model biner ensemble tersimpan (.pkl)
│   └── submission_final.csv                  # Hasil prediksi data uji
├── generate_data.py             # Script generator dataset simulasi
├── train_baseline.py            # Script pipeline pelatihan otomatis
├── FINAL_PRESENTATION_DECK.md   # Panduan slide presentasi proyek (Pitch Deck 10 Menit)
├── TECHNICAL_REPORT.md          # Draf laporan teknis metodologi
├── MODELING_GUIDE.md            # Panduan eksperimen, tuning, dan menambah model
├── requirements.txt             # Dependensi Python
├── .gitignore                   # Konfigurasi file terabaikan Git
└── README.md                    # Dokumentasi portofolio profesional
```

---

## 🚀 Panduan Menjalankan Proyek
1. Clone repositori:
   ```bash
   git clone https://github.com/ayubwicak/exam-cheating-detection-ml.git
   cd exam-cheating-detection-ml
   ```
2. Buat virtual environment dan pasang dependensi:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Jalankan notebook interaktif di VS Code / Jupyter:
   ```bash
   code notebooks/01_cheating_detection_solution.ipynb
   ```

   Notebook disimpan tanpa output eksekusi agar ukuran repo tetap ringan.
   Jalankan ulang sel dari atas untuk melihat tabel dan plot.
4. Jalankan turnamen multi-model (opsional, tetapi disarankan):
   ```bash
   python -m src.train --models all --feature-sets A,B,C
   ```

   Perintah di atas membandingkan beberapa algoritma dan feature set dengan
   Stratified K-Fold dan threshold tuning. Untuk repeated CV + confidence
   interval, gunakan model yang lebih ringan agar tidak terlalu lama:
   ```bash
   python -m src.train --models logreg,hist_gb,lightgbm,xgboost \
     --feature-sets A,B,C --repeats 3 --bootstrap 200
   ```
   Hasilnya disimpan di `output/tournament_results.csv`,
   `output/best_model_bundle.pkl`, dan `output/submission_tournament.csv`.
   Panduan lengkap untuk tuning dan menambah model ada di
   [`MODELING_GUIDE.md`](MODELING_GUIDE.md).

> **Catatan:** dataset pada repositori ini adalah data simulasi. Skor
> ROC-AUC yang sangat tinggi mencerminkan aturan generator yang dapat
> dipelajari model, bukan performa dunia nyata.
