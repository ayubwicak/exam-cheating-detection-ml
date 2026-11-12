# 🎯 RANCANGAN SLIDE PRESENTASI PROYEK (10 MENIT)
**Topik:** AI-Powered Online Exam Cheating Detection (CBT Anomaly Detection)  
**Tujuan:** Presentasi Studi Kasus / Project Showcase Data Science  
**Alokasi Waktu:** 10 Menit Presentasi + 15 Menit Tanya Jawab (Q&A)

---

## 📊 Distribusi Slide & Timeline 10 Menit

| No. Slide | Judul Slide | Durasi Bicara | Fokus Penilaian Juri |
| :---: | :--- | :---: | :--- |
| **1** | **Judul & Perkenalan Tim** | 0.5 Menit | Sikap profesional & kejelasan identitas tim |
| **2** | **Latar Belakang & Urgensi Masalah (SDGs 4)** | 1.5 Menit | Analisis Masalah (Bobot 25%) |
| **3** | **Exploratory Data Analysis (EDA) & Anomali** | 2.0 Menit | Analisis Masalah & Kedalaman Pemahaman |
| **4** | **Formulasi Hipotesis & Feature Engineering** | 2.0 Menit | Kreativitas Solusi & Domain Knowledge |
| **5** | **Strategi Pemodelan & Validasi (Anti-Leakage)** | 1.5 Menit | Kedalaman Pemahaman Teknis (Bobot 20%) |
| **6** | **Hasil Turnamen Model & Metrik Evaluasi** | 1.0 Menit | Pembuktian Efektivitas Model |
| **7** | **Interpretasi Model (Explainable AI / XAI)** | 1.0 Menit | Transparansi & Akuntabilitas Sistem |
| **8** | **Kesimpulan & Rekomendasi Implementasi CBT** | 0.5 Menit | Dampak nyata dan aplikatif |

---

## 📝 Detail Isi per Slide

### Slide 1: Cover / Judul Proyek
* **Judul:** AI-Driven Exam Integrity: Deteksi Kecurangan Ujian Berbasis Gradient Boosted Decision Trees dan Analisis Perilaku Multidimensi
* **Sub-judul:** Mendukung Integritas Pendidikan Sesuai SDGs 4 (Quality Education)
* **Informasi Tim:** Nama Tim, Nama Anggota (Presenter 1 & Presenter 2), Asal Universitas.

### Slide 2: Latar Belakang & Tantangan Nyata (Problem Statement)
* **Tantangan:** Era CBT mempermudah evaluasi massal, namun memunculkan celah kecurangan: browsing jawaban di tab lain, copy-paste soal ke forum/LLM, hingga penggunaan joki via remote desktop.
* **Kelemahan Sistem Konvensional:** Deteksi manual berbasis kamera proctoring memakan bandwidth besar dan rentan bias subjektif.
* **Solusi Kami:** Sistem deteksi berbasis data log telemetri yang ringan, objektif, dan bekerja secara real-time.

### Slide 3: Temuan Exploratory Data Analysis (EDA)
* Tampilkan **Grafik Sebaran (Scatter Plot)**: Mayoritas peserta curang memiliki sebaran durasi ujian yang tidak wajar terhadap skor akhirnya.
* Tampilkan **Grafik Anomali IP vs Tab Switch**: Hubungan kuat antara perubahan IP dinamis dengan lonjakan aktivitas keluar tab browser.
* *Insight Kunci:* Kasus curang merupakan fenomena *imbalanced class* (~10-15%), menuntut strategi pembobotan khusus agar model tidak bias ke kelas mayoritas.

### Slide 4: Feature Engineering (Inovasi Utama Tim)
* **Ide 1: Speedrun Anomaly**
  $$\text{Score Speed Ratio} = \frac{\text{Score}}{\text{Duration (min)}}$$
  *Siswa dengan rasio skor per menit ekstrem tinggi diisolasi melalui indikator biner.*
* **Ide 2: Joki Remote Anomaly**
  $$\text{Joki Anomaly Score} = \text{Network IP Changes} \times \text{Tab Switch Count}$$
  *Kombinasi penggantian jaringan dengan perpindahan window aktif menangkap jejak digital software remote desktop / joki.*

### Slide 5: Metodologi Pemodelan & Validasi Kokoh
* **Cross-Validation:** 5-Fold Stratified CV (menjaga proporsi kelas target merata di setiap fold).
* **Reproducibility:** Seluruh modul dikunci dengan random seed 42 (mematuhi regulasi ketat halaman 6 poin 9).
* **Arsitektur:** LightGBM Classifier dengan penanganan *cost-sensitive learning* (`class_weight='balanced'`).
* **Ensembling:** Soft voting averaging dari kelima fold model untuk stabilitas inferensi test data.

### Slide 6: Hasil Komparasi & Kemenangan Model
* Tampilkan Tabel Turnamen:
  * Baseline Awal: F1-Score **85.79%**
  * Baseline + Ide 1: F1-Score **85.52%**
  * Baseline + Ide 1 & 2 (Model C): F1-Score **87.74%** (**Kenaikan +1.95%**)
  * ROC-AUC: **0.9941** (Diskriminasi kelas hampir sempurna).

### Slide 7: Explainable AI (XAI) & Akuntabilitas
* Tampilkan grafik **Feature Importance**.
* *Pesan ke Juri:* *"Model kami bukan black-box! Kami dapat menunjukkan kepada dosen/pengawas bukti spesifik mengapa seorang mahasiswa dicurigai curang (misal: bobot terbesar datang dari lonjakan joki anomaly dan score-speed ratio)."*

### Slide 8: Kesimpulan & Implementasi Nyata
* **Kesimpulan:** Model C berhasil memberikan deteksi presisi tinggi tanpa membebani infrastruktur server ujian.
* **Rekomendasi Praktis:** Integrasi langsung ke dashboard proctoring CBT kampus berupa *risk score meter* (0 - 100%) untuk menandai mahasiswa yang perlu diawasi lebih ketat.

---

## 🎯 Panduan Menghadapi Sesi Tanya Jawab (Q&A - Bobot 30%)
* **Pertanyaan Juri:** *"Kenapa memilih F1-score bukan akurasi saja?"*  
  **Jawaban:** Karena data kecurangan sangat timpang (*imbalanced*). Jika model selalu menebak jujur, akurasi tetap 90%, tetapi gagal total menangkap pencurang. F1-score menjamin keseimbangan antara *precision* (tidak salah tuduh) dan *recall* (menangkap semua pencurang).
* **Pertanyaan Juri:** *"Kenapa memakai LightGBM bukan Deep Learning (Neural Network)?"*  
  **Jawaban:** Untuk data tabular terstruktur berukuran ribuan baris, algoritma berbasis *Gradient Boosted Decision Trees* terbukti secara empiris lebih cepat konvergen, tahan overfitting, dan interpretable dibanding neural network.
