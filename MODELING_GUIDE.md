# Panduan Modeling & Eksperimen Lanjutan

Dokumen ini merangkum penilaian pipeline saat ini, cara mengasah model,
dan cara menambahkan model/feature set baru tanpa mengubah notebook.

---

## 1. Penilaian kondisi saat ini

### Yang sudah baik
- Pipeline end-to-end sudah jelas: EDA → feature engineering → 5-fold
  Stratified CV → evaluasi → penyimpanan model → submission.
- Ada hipotesis domain (speedrun cheating dan joki/remote access), bukan
  sekadar "lempar semua fitur ke model".
- `class_weight="balanced"` sudah dipakai, sehingga kelas minoritas tidak
  diabaikan sepenuhnya.
- Random seed dikunci dan hasil eksperimen dapat direproduksi.
- Model disimpan dalam `.pkl` dan submission dihasilkan otomatis.

### Catatan penting: skor tinggi berasal dari data sintetis
Dataset dibuat oleh `generate_data.py`. Label `is_cheating` dihitung
langsung dari fitur-fitur yang sama yang dipakai model:

```text
cheat_score = 0.4 * tab_switch
            + 0.35 * copy_paste
            + 4.0 * similarity
            + 1.2 * ip_changes
            + 3.0 * (score > 85 & duration < 35)
            - 1.5 * mouse_entropy
            - 3.2
```

Karena itu ROC-AUC ≈ 0.99 **bukan bukti bahwa model akan sehebat itu di
dunia nyata**. Model hanya mempelajari aturan generator. Untuk portofolio,
ini tetap sah sebagai studi kasus simulasi, tetapi harus ditulis sebagai
"hasil pada data sintetis", bukan klaim performa dunia nyata.

### Kelemahan teknis yang perlu diperbaiki
1. **Hanya satu kali 5-fold CV.** Hasil bisa berubah cukup besar hanya
   karena split. Gunakan repeated CV (misalnya 3–5 pengulangan) dan
   laporkan rata-rata ± standar deviasi.
2. **F1 dihitung pada threshold 0.5.** Dengan proporsi curang hanya
   sekitar 9,94%, threshold 0.5 belum tentu optimal. Threshold harus
   dicari dari data validasi/OOF.
3. **Belum ada PR-AUC.** Untuk kelas tidak seimbang, PR-AUC
   (`average_precision_score`) lebih informatif daripada ROC-AUC.
4. **Belum ada baseline sederhana.** Logistic Regression pada fitur
   mentah justru bisa mengalahkan LightGBM di dataset ini. Baseline
   penting untuk memastikan model kompleks benar-benar memberi nilai
   tambah.
5. **Risiko multikolinearitas.** `avg_time_per_question` di generator
   sama dengan `duration_minutes * 60 / 50`, jadi informasinya redundan
   dengan durasi.
6. **Model yang disimpan tidak konsisten.** Notebook menyimpan
   `models_C[0]` (satu fold), tetapi prediksi memakai rata-rata lima
   fold. Bundle baru menyimpan seluruh model fold.
7. **Belum ada kalibrasi probabilitas, analisis error, dan pengujian
   robustness.** ROC-AUC tinggi tidak menjamin probabilitas model
   terkalibrasi.
8. **Belum ada requirements, test, dan registry model.** Semua logika
   masih tersebar di notebook dan dua script.

### Temuan dari turnamen model
Dengan repeated 3×5-fold CV (seed 42, 43, 44) dan threshold yang
dioptimalkan pada OOF:

| Model | Feature set | ROC-AUC | PR-AUC | F1 terbaik |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | A | 0.9900 ± 0.0003 | **0.9719 ± 0.0001** | 0.9244 ± 0.0046 |
| Logistic Regression | B | 0.9931 ± 0.0010 | 0.9358 ± 0.0055 | **0.9311 ± 0.0042** |
| LightGBM | C | **0.9940 ± 0.0012** | 0.9593 ± 0.0077 | 0.8899 ± 0.0096 |
| HistGradientBoosting | C | 0.9939 ± 0.0002 | 0.9567 ± 0.0019 | 0.8852 ± 0.0044 |

Angka di atas berasal dari eksperimen repeated CV yang sama dengan
`python -m src.train --models all --feature-sets A,B,C --repeats 3`.
Kesimpulannya:

- Jika metrik utama **PR-AUC**, Logistic Regression dengan fitur mentah
  (set A) menang. Feature engineering justru menambah noise.
- Jika metrik utama **ROC-AUC**, LightGBM dengan set C sedikit lebih baik.
- Jika metrik utama **F1 pada threshold optimal**, Logistic Regression
  dengan set B menang tipis (0.9311 vs 0.9244 untuk set A), sedangkan
  LightGBM C hanya 0.8899.
- Klaim "Model C selalu terbaik" tidak cukup kuat. Yang benar:
  **LightGBM C unggul pada ROC-AUC, tetapi Logistic Regression unggul
  pada PR-AUC dan F1 setelah threshold tuning.**

Ini bukan berarti feature engineering buruk. Artinya, setiap fitur baru
harus diuji dengan metrik yang tepat dan repeated CV, bukan diasumsikan
selalu meningkatkan performa.

---

## 2. Cara mengasah model

### 2.1 Pilih metrik sesuai tujuan
Gunakan beberapa metrik sekaligus:

| Metrik | Kapan dipakai |
| :--- | :--- |
| ROC-AUC | Ranking umum, tetapi bisa terlihat tinggi pada data imbalanced |
| PR-AUC / Average Precision | Lebih sensitif ke kelas minoritas (curang) |
| F1 pada threshold optimal | Jika keputusan akhir biner dan F1 adalah target |
| Recall pada precision ≥ 95% | Jika false positive sangat mahal |
| Brier score / log-loss | Mengukur kualitas probabilitas |
| Confusion matrix | Melihat jenis kesalahan (FP vs FN) |

Jangan memilih model hanya dari satu angka. Laporkan juga confidence
interval, misalnya dengan bootstrap.

### 2.2 Tuning threshold
Threshold 0.5 adalah default, bukan hukum. Cari threshold yang
memaksimalkan F1, atau yang memenuhi syarat bisnis:

```python
from src.evaluation import find_best_threshold

threshold, best_f1 = find_best_threshold(y_oof, oof_probabilities)
```

Simpan threshold bersama model. Saat inference, gunakan threshold yang
sama. Perhatikan bahwa memilih threshold pada OOF lalu melaporkan F1 pada
OOF yang sama sedikit optimistis; untuk laporan final, gunakan nested CV
atau set kalibrasi terpisah.

### 2.3 Repeated CV dan confidence interval
Satu split 5-fold bisa beruntung atau sial. Jalankan beberapa seed:

```bash
python -m src.train \
  --models logreg,hist_gb,lightgbm,xgboost \
  --feature-sets A,B,C \
  --repeats 3 \
  --bootstrap 200
```

Script akan menampilkan `mean ± std` untuk ROC-AUC, PR-AUC, dan F1.
Prioritaskan model yang stabil (std kecil), bukan hanya yang menang tipis.
Random Forest/ExtraTrees dengan 500 pohon bisa lambat jika diulang
berkali-kali; batasi daftar model saat melakukan repeated CV.

### 2.4 Hyperparameter tuning
Setelah memilih 2–3 kandidat terbaik, lakukan tuning yang lebih serius.
Contoh dengan `RandomizedSearchCV`:

```python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from src.models import make_model

search = RandomizedSearchCV(
    estimator=make_model("lightgbm"),
    param_distributions={
        "n_estimators": [200, 300, 500, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "max_depth": [-1, 3, 5, 7],
        "min_child_samples": [5, 10, 20, 40],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    },
    n_iter=40,
    scoring="average_precision",
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
```

Catatan: tuning harus dilakukan **di dalam** training fold jika ingin
estimasi performa yang benar-benar jujur (nested CV). Jika tidak, skor
tuning bisa bocor ke data evaluasi.

### 2.5 Ensemble
Coba gabungkan model yang berbeda karakter:

- LightGBM + XGBoost + CatBoost (gradient boosting)
- Logistic Regression + tree model (linear + non-linear)
- Soft voting: rata-rata probabilitas
- Stacking: meta-model Logistic Regression di atas prediksi OOF

Ensemble tidak selalu menang. Uji dengan repeated CV dan bandingkan
terhadap model tunggal terbaik.

### 2.6 Feature engineering yang lebih disiplin
- Buang atau ganti fitur yang redundan seperti `avg_time_per_question`
  jika sudah ada `duration_minutes`.
- Tambahkan fitur berbasis domain: `tab_switch_per_minute`,
  `copy_paste_per_minute`, `ip_change_per_minute`, time pressure.
- Uji setiap kelompok fitur secara terpisah (ablasi fitur), seperti
  Model A → B → C, tetapi dengan repeated CV dan PR-AUC.
- Gunakan permutation importance atau SHAP untuk melihat apakah fitur
  benar-benar dipakai model.
- Waspadai fitur yang hanya ada setelah ujian selesai (misalnya skor
  akhir) jika model ingin dipakai untuk deteksi real-time. Untuk
  deteksi setelah ujian, fitur tersebut boleh; untuk deteksi saat ujian
  berlangsung, tidak boleh.

### 2.7 Kalibrasi probabilitas
Model tree biasanya kurang terkalibrasi. Coba:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(
    estimator=make_model("lightgbm"),
    method="isotonic",
    cv=5,
)
```

Evaluasi dengan Brier score, log-loss, dan reliability diagram. Ini
penting jika probabilitas dipakai untuk prioritas pemeriksaan manual.

### 2.8 Analisis error dan robustness
- Lihat false positive dan false negative terbanyak.
- Bandingkan error berdasarkan kelompok skor, durasi, dan perubahan IP.
- Uji dengan noise pada fitur (misalnya 5% data hilang atau fitur
  diacak) untuk melihat ketahanan model.
- Uji dengan data dari periode/semester berbeda jika tersedia.
- Diskusikan fairness: jangan sampai fitur proksi (misalnya jenis
  perangkat atau koneksi) menimbulkan bias yang tidak diinginkan.

---

## 3. Menjalankan turnamen model

### Instalasi
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Perintah dasar
```bash
# Semua model pada feature set A, B, C
python -m src.train --models all --feature-sets A,B,C

# Model tertentu saja
python -m src.train --models lightgbm,xgboost --feature-sets C

# Repeated CV + bootstrap confidence interval
python -m src.train \
  --models logreg,hist_gb,lightgbm,xgboost \
  --feature-sets A,B,C \
  --repeats 3 \
  --bootstrap 200

# Pilih model berdasarkan F1 optimal, bukan PR-AUC
python -m src.train --models all --feature-sets A,B,C --select-metric best_f1
```

### Output
```text
output/tournament_results.csv    # tabel semua kombinasi model + fitur
output/best_model_bundle.pkl     # semua model fold + fitur + threshold
output/best_model_metrics.json   # metrik model terbaik
output/submission_tournament.csv # submission dari model terbaik
```

### Inference
```bash
python -m src.predict \
  --input data/test.csv \
  --bundle output/best_model_bundle.pkl \
  --output output/submission.csv
```

---

## 4. Cara menambahkan model baru

Semua model terdaftar di `src/models.py` melalui decorator
`@register_model`. Pipeline `src/train.py` otomatis menemukan model baru
tanpa perlu mengubah loop pelatihan.

### Langkah 1: tambahkan factory
Contoh menambahkan CatBoost:

```python
# src/models.py
@register_model("catboost")
def build_catboost(
    *, seed: int = SEED, pos_weight: float = 1.0, **kwargs
):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=5,
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=False,
        **kwargs,
    )
```

### Langkah 2: tambahkan ke daftar default (opsional)
Jika ingin ikut berjalan saat `--models all`:

```python
DEFAULT_MODELS = [
    "logreg",
    "random_forest",
    "extra_trees",
    "hist_gb",
    "lightgbm",
    "xgboost",
    "catboost",  # tambahkan di sini
]
```

### Langkah 3: jalankan
```bash
pip install catboost
python -m src.train --models catboost --feature-sets C
```

Model baru langsung mengikuti protokol yang sama: Stratified K-Fold,
threshold tuning, metrik, dan penyimpanan bundle.

### Contoh model wrapper
Jika ingin memakai model yang bukan classifier scikit-learn, bungkus
agar memiliki `fit` dan `predict_proba`:

```python
class ModelWrapper:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
```

---

## 5. Cara menambahkan feature set baru

Edit `src/features.py`:

```python
FEATURE_SETS["D"] = tuple(
    BASE_FEATURES
    + SPEED_FEATURES
    + JOKI_FEATURES
    + ACTIVITY_FEATURES
    + ["tab_switch_per_minute", "copy_paste_per_minute"]
)
```

Lalu pastikan fitur tersebut dibuat di `add_domain_features`:

```python
out["tab_switch_per_minute"] = out["tab_switch_count"] / duration
out["copy_paste_per_minute"] = out["copy_paste_count"] / duration
```

Jalankan:

```bash
python -m src.train --models lightgbm --feature-sets D
```

Selalu bandingkan dengan feature set sebelumnya. Jika PR-AUC/F1 tidak
naik secara konsisten pada repeated CV, fitur baru belum layak dipakai.

---

## 6. Checklist sebelum submit

- [ ] Gunakan repeated CV (minimal 3 seed) dan laporkan mean ± std.
- [ ] Laporkan ROC-AUC, PR-AUC, F1 pada threshold optimal, precision,
      recall, dan confusion matrix.
- [ ] Tentukan threshold dari data validasi/OOF, bukan dari data test.
- [ ] Bandingkan dengan baseline Dummy dan Logistic Regression.
- [ ] Cek multikolinearitas dan fitur yang berpotensi leakage.
- [ ] Uji stabilitas fitur dengan permutation importance/SHAP.
- [ ] Simpan seluruh model fold, bukan hanya satu fold.
- [ ] Simpan threshold dan daftar fitur di dalam bundle.
- [ ] Jalankan ulang dari environment bersih dengan `requirements.txt`.
- [ ] Tulis dengan jelas bahwa dataset adalah simulasi/sintetis.

---

## 7. Batasan dan langkah realistis berikutnya

Untuk dataset sintetis ini, skor sudah mendekati plafon. Peningkatan
terbesar bukan dari menambah model yang lebih rumit, melainkan dari:

1. **Memperbaiki protokol evaluasi** (repeated CV, PR-AUC, threshold).
2. **Membuat data benchmark lebih realistis** (label noise, overlap,
   missing values, distribution shift).
3. **Membandingkan baseline dan model kompleks secara adil.**
4. **Menulis kesimpulan yang jujur** tentang keterbatasan data sintetis.

Jika ingin meningkatkan kredibilitas portofolio, pertimbangkan dataset
publik nyata (misalnya log ujian atau data perilaku mahasiswa), lalu
terapkan pipeline yang sama. Di sana, penurunan performa dari 0.99 ke
0.75–0.85 justru merupakan hasil yang lebih bernilai dan lebih dipercaya.
