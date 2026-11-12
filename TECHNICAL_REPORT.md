# 📄 LAPORAN TEKNIS (TECHNICAL REPORT)
### Data Slayer 4.0: Machine Learning Competition
**Subtema:** *"Honest Score, Equal Chance: Exam Cheating Detection"*

---

## 1. Pendahuluan
Integritas akademik adalah pilar utama penilaian mutu pendidikan (SDGs 4: Quality Education). Pelaksanaan Computer-Based Test (CBT) skala besar memiliki kerentanan berupa kecurangan asinkron yang sulit dideteksi secara manual oleh pengawas. Laporan teknis ini menguraikan perancangan sistem Machine Learning end-to-end yang mengombinasikan rekayasa fitur domain-spesifik dan algoritma *Light Gradient Boosting Machine (LightGBM)* untuk mendeteksi anomali kecurangan dengan presisi tinggi.

---

## 2. Eksplorasi Data & Identifikasi Anomali (EDA)
Dataset terdiri dari 3.500 baris data latih (*train*) dan 1.500 baris data uji (*test*). Analisis eksploratif menunjukkan karakteristik utama data:
1. **Ketimpangan Kelas (Class Imbalance):** Kasus kecurangan hanya mencakup ~10% dari total populasi ujian, menuntut skema pembobotan sampel pada saat pelatihan.
2. **Korelasi Multivariat:** Pelaku kecurangan menunjukkan deviasi ekstrem pada dua pola:
   * Korelasi terbalik antara waktu penyelesaian dengan capaian skor.
   * Tingginya frekuensi perpindahan aplikasi/tab yang bertepatan dengan perubahan IP koneksi jaringan.

---

## 3. Rekayasa Fitur (Feature Engineering)
Dua fitur anomali utama diformulasikan untuk memperkuat daya diskriminasi model:

1. **Rasio Kecepatan Perolehan Nilai (`score_speed_ratio`)**
   $$\text{score\_speed\_ratio} = \frac{\text{score}}{\text{duration\_minutes} + \epsilon}$$
   Mengidentifikasi mahasiswa yang menyelesaikan soal secara instan namun dengan perolehan skor sempurna, mengindikasikan kebocoran kunci jawaban sebelumnya.

2. **Skor Anomali Joki Remote (`joki_remote_anomaly`)**
   $$\text{joki\_remote\_anomaly} = \text{network\_ip\_changes} \times \text{tab\_switch\_count}$$
   Menangkap jejak peralihan akses ujian ke pihak ketiga melalui software remote desktop atau pergantian rute proxy/VPN.

---

## 4. Metodologi Pemodelan & Validasi
Untuk mencegah kebocoran data (*data leakage*) dan memastikan reproduktifitas sesuai aturan panitia:
* **Stratified 5-Fold Cross Validation:** Digunakan untuk memvalidasi model secara lokal dengan menjaga rasio target seimbang di setiap fold.
* **Penguncian Random Seed:** Nilai seed `42` diterapkan secara seragam pada seluruh modul acak.
* **Pemilihan Algoritma:** LightGBM Classifier dipilih karena efisiensinya dalam menangani data tabular, kecepatan pelatihan, serta dukungan terhadap bobot kelas otomatis (`class_weight='balanced'`).

---

## 5. Hasil & Evaluasi
Eksperimen bertahap menunjukkan peningkatan performa signifikan:

| Konfigurasi Model | Out-of-Fold ROC-AUC | Out-of-Fold F1-Score | Keterangan |
| :--- | :---: | :---: | :--- |
| **Model A (Baseline Mentah)** | 0.9930 | 85.79% | Fitur standar |
| **Model B (+ Ide 1: Speedrun)** | 0.9929 | 85.52% | Penambahan rasio nilai |
| **Model C (+ Ide 1 & 2: Joki)** | **0.9941** | **87.74%** | **Model Final (+1.95% F1)** |

---

## 6. Interpretasi Model & Rekomendasi
Analisis *Feature Importance* menegaskan bahwa fitur anomali joki (`joki_remote_anomaly`) dan rasio waktu-nilai (`score_speed_ratio`) menduduki peringkat teratas dalam proses klasifikasi. Solusi ini siap diintegrasikan pada arsitektur CBT skala besar sebagai *early-warning system* pengawasan ujian yang adil dan transparan.
