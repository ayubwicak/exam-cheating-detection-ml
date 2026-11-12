import numpy as np
import pandas as pd

# Set random seed untuk reproducibility sesuai aturan lomba
np.random.seed(42)

def generate_exam_data(n_samples=4000, is_test=False):
    student_ids = [f"STD_{10000 + i if not is_test else 50000 + i}" for i in range(n_samples)]
    
    # 1. Tab switch count (orang curang biasanya sering pindah tab untuk cari jawaban)
    tab_switch = np.random.poisson(lam=2, size=n_samples)
    
    # 2. Copy paste count (copy pertanyaan atau paste jawaban)
    copy_paste = np.random.poisson(lam=1.5, size=n_samples)
    
    # 3. Exam duration in minutes (total waktu ujian 90 menit)
    duration = np.random.normal(loc=65, scale=15, size=n_samples)
    duration = np.clip(duration, 15, 90)
    
    # 4. Average time per question (dalam detik, misal ada 50 soal)
    avg_time = (duration * 60) / 50
    
    # 5. Network IP changes (berganti wifi/tethering/joki remote)
    ip_changes = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.85, 0.10, 0.04, 0.01])
    
    # 6. Mouse movement entropy (0.0 kaku - 1.0 sangat dinamis)
    mouse_entropy = np.random.beta(a=5, b=2, size=n_samples)
    
    # 7. Similarity with peers (kemiripan urutan dan jawaban dengan siswa lain)
    similarity = np.random.beta(a=2, b=5, size=n_samples)
    
    # 8. Score akhir (0 - 100)
    base_score = np.random.normal(loc=68, scale=14, size=n_samples)
    score = np.clip(np.round(base_score), 20, 100).astype(int)
    
    df = pd.DataFrame({
        "student_id": student_ids,
        "duration_minutes": np.round(duration, 2),
        "score": score,
        "tab_switch_count": tab_switch,
        "copy_paste_count": copy_paste,
        "avg_time_per_question": np.round(avg_time, 2),
        "mouse_movement_entropy": np.round(mouse_entropy, 3),
        "network_ip_changes": ip_changes,
        "similarity_with_peers": np.round(similarity, 3)
    })
    
    if not is_test:
        # Simulasi probabilitas curang berdasarkan pola log
        # Logika: curang jika tab switch tinggi, copy paste tinggi, similarity tinggi, ip ganti, atau skor tinggi tapi waktu sangat singkat
        cheat_score = (
            (tab_switch * 0.4) +
            (copy_paste * 0.35) +
            (similarity * 4.0) +
            (ip_changes * 1.2) +
            ((score > 85) & (duration < 35)).astype(int) * 3.0 -
            (mouse_entropy * 1.5) - 3.2
        )
        # Sigmoid untuk probabilitas
        prob = 1 / (1 + np.exp(-cheat_score))
        # Label biner (0 = jujur, 1 = curang), sekitar 15-20% curang
        df["is_cheating"] = (prob > 0.5).astype(int)
        
    return df

# Generate Train (3,500 data) & Test (1,500 data)
train_df = generate_exam_data(n_samples=3500, is_test=False)
test_df = generate_exam_data(n_samples=1500, is_test=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

# Sample submission
sample_sub = pd.DataFrame({
    "student_id": test_df["student_id"],
    "is_cheating": 0
})
sample_sub.to_csv(os.path.join(DATA_DIR, "sample_submission.csv"), index=False)

print(f"Data Berhasil Dibuat!")
print(f"Train Shape: {train_df.shape}, Distribusi Target: {dict(train_df['is_cheating'].value_counts())}")
print(f"Test Shape: {test_df.shape}")
