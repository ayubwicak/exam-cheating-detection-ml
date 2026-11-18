"""Aplikasi Web Dashboard Interaktif - AI Exam Proctoring & Anomaly Detection.

Jalankan dengan:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from src.config import OUTPUT_DIR, SEED, TRAIN_PATH
from src.features import build_feature_frame
from src.models import make_model, available_models

st.set_page_config(
    page_title="Exam Cheating Detection AI",
    page_icon="🎓",
    layout="wide",
)

# Custom Styling CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 AI-Powered Online Exam Proctoring Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Sistem evaluasi integritas ujian online CBT berbasis Machine Learning & Explainable AI (XAI)</div>',
    unsafe_allow_html=True
)

# ==========================================
# SIDEBAR: PENGATURAN MODEL & AMBANG BATAS
# ==========================================
st.sidebar.header("⚙️ Konfigurasi Sistem")

model_choice = st.sidebar.selectbox(
    "Pilihan Algoritma Machine Learning",
    options=["lightgbm", "xgboost", "hist_gb", "random_forest", "logreg"],
    index=0,
    help="Pilih algoritma pemodelan yang ingin digunakan untuk deteksi.",
)

feature_choice = st.sidebar.selectbox(
    "Pilihan Feature Set",
    options=[
        "D (Terlengkap: Baseline + Speedrun + Joki + Collusion Interaksi)",
        "C (Baseline + Speedrun + Joki)",
        "B (+ Speedrun Anomaly)",
        "A (Baseline Mentah)"
    ],
    index=0,
    help="Feature Set D menambahkan interaksi kolusi kemiripan rekan dengan clipboard dan tab switch.",
)
fset_key = feature_choice[0]

threshold = st.sidebar.slider(
    "Ambang Batas Vonis (Decision Threshold)",
    min_value=0.20,
    max_value=0.90,
    value=0.54,
    step=0.02,
    help="Standar optimal F1-score kompetisi adalah 0.54 (keseimbangan Precision & Recall).",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Panduan Kebijakan Pengawas")
if threshold < 0.45:
    st.sidebar.warning("🛡️ **Mode Sangat Ketat:** Recall tinggi. Meminimalkan peserta curang lolos, namun risiko *false positive* (salah tuduh) sedikit meningkat.")
elif threshold <= 0.65:
    st.sidebar.success("⚖️ **Mode Seimbang (Optimal):** Titik tengah F1-Score terbaik untuk meminimalisasi salah vonis sekaligus menangkap mayoritas kecurangan.")
else:
    st.sidebar.info("🎯 **Mode Konservatif (Hati-Hati):** Presisi sangat tinggi. Hanya menandai peserta jika bukti telemetri mutlak/sangat kuat.")


# Load data & latih model jika belum ada di session state
@st.cache_resource
def get_trained_model(selected_model: str, selected_fset: str):
    train_df = pd.read_csv(TRAIN_PATH)
    X = build_feature_frame(train_df, selected_fset)
    if isinstance(X, tuple):
        X = X[0]
    y = train_df["is_cheating"]
    model = make_model(selected_model, seed=SEED)
    model.fit(X, y)
    return model, X.columns.tolist()

with st.spinner(f"Menyiapkan model {model_choice.upper()} (Feature Set {fset_key})..."):
    model, feature_cols = get_trained_model(model_choice, fset_key)

# ==========================================
# INISIALISASI SESSION STATE UNTUK PRESET
# ==========================================
default_presets = {
    "score": 95,
    "duration": 20.0,
    "avg_time": 24.0,
    "tab_switches": 3,
    "copy_pastes": 2,
    "ip_changes": 1,
    "mouse_entropy": 0.75,
    "peer_similarity": 0.20,
}

for key, val in default_presets.items():
    if key not in st.session_state:
        st.session_state[key] = val


def apply_preset(score: int, dur: float, avg: float, tab: int, cp: int, ip: int, mouse: float, peer: float):
    st.session_state["score"] = score
    st.session_state["duration"] = dur
    st.session_state["avg_time"] = avg
    st.session_state["tab_switches"] = tab
    st.session_state["copy_pastes"] = cp
    st.session_state["ip_changes"] = ip
    st.session_state["mouse_entropy"] = mouse
    st.session_state["peer_similarity"] = peer


# ==========================================
# TAB LAYOUT
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Audit Mahasiswa Tunggal",
    "📊 Transparansi AI (SHAP)",
    "📁 Penilaian Massal (Batch CSV)",
    "🏆 Papan Skor Model (Tournament)"
])

# ------------------------------------------
# TAB 1: AUDIT MAHASISWA TUNGGAL
# ------------------------------------------
with tab1:
    st.subheader("Simulasi Audit Telemetri Mahasiswa")
    st.markdown("Pilih profil skenario cepat atau masukkan parameter telemetri manual:")

    # Tombol Preset dengan callback yang persisten
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.button(
            "👨‍🎓 1. Siswa Wajar (Normal)",
            on_click=apply_preset,
            args=(92, 65.0, 78.0, 1, 0, 0, 0.85, 0.15),
            width="stretch",
            help="Siswa belajar rajin, nilai tinggi dengan durasi wajar tanpa anomali tab/IP.",
        )
    with col_p2:
        st.button(
            "⚡ 2. Speedrun Cheater",
            on_click=apply_preset,
            args=(98, 18.0, 21.6, 2, 0, 0, 0.80, 0.20),
            width="stretch",
            help="Selesai dalam waktu kilat 18 menit dengan skor 98.",
        )
    with col_p3:
        st.button(
            "🚨 3. Pengguna Joki / Remote",
            on_click=apply_preset,
            args=(88, 70.0, 84.0, 8, 4, 2, 0.55, 0.25),
            width="stretch",
            help="Pergantian IP di tengah ujian disertai tab switch dan aktivitas copy-paste tinggi.",
        )
    with col_p4:
        st.button(
            "📶 4. Kendala Koneksi (Lag)",
            on_click=apply_preset,
            args=(74, 75.0, 90.0, 7, 0, 0, 0.95, 0.12),
            width="stretch",
            help="Siswa keluar masuk tab karena koneksi lemot, namun tidak ada duplikasi jawaban/IP.",
        )

    st.markdown("---")

    # Form input telemetri
    col1, col2, col3 = st.columns(3)
    with col1:
        score = st.number_input(
            "Nilai Ujian (Score)",
            min_value=0,
            max_value=100,
            key="score",
            help="Skor perolehan ujian akhir (0 - 100)."
        )
        duration = st.number_input(
            "Durasi Pengerjaan (Menit)",
            min_value=1.0,
            max_value=180.0,
            step=1.0,
            key="duration",
            help="Waktu dari mulai mengerjakan sampai submit (menit)."
        )
        avg_time = st.number_input(
            "Rata-rata Waktu per Soal (Detik)",
            min_value=1.0,
            max_value=300.0,
            step=1.0,
            key="avg_time",
            help="Durasi rata-rata yang dihabiskan siswa per butir soal (detik)."
        )

    with col2:
        tab_switches = st.number_input(
            "Frekuensi Pindah Tab Browser",
            min_value=0,
            max_value=100,
            step=1,
            key="tab_switches",
            help="Berapa kali siswa berpindah jendela/tab browser selama ujian berlangsung."
        )
        copy_pastes = st.number_input(
            "Frekuensi Copy-Paste",
            min_value=0,
            max_value=100,
            step=1,
            key="copy_pastes",
            help="Jumlah operasi salin-tempel teks di lembar jawaban ujian."
        )
        ip_changes = st.number_input(
            "Pergantian Alamat IP Jaringan",
            min_value=0,
            max_value=20,
            step=1,
            key="ip_changes",
            help="Jumlah pergantian IP address jaringan selama sesi ujian aktif."
        )

    with col3:
        mouse_entropy = st.slider(
            "Aktivitas Kursor Mouse (Entropy)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="mouse_entropy",
            help="Kerapatan dan keacakan gerakan kursor mouse. Nilai rendah menandakan pasif/bot/script."
        )
        peer_similarity = st.slider(
            "Kemiripan Jawaban dengan Rekan (Similarity)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="peer_similarity",
            help="Tingkat kemiripan pola respon/jawaban dengan mahasiswa lain (0.0 - 1.0)."
        )

    # Prediksi menggunakan pipeline
    input_data = pd.DataFrame([{
        "duration_minutes": duration,
        "score": score,
        "tab_switch_count": tab_switches,
        "copy_paste_count": copy_pastes,
        "avg_time_per_question": avg_time,
        "mouse_movement_entropy": mouse_entropy,
        "network_ip_changes": ip_changes,
        "similarity_with_peers": peer_similarity,
    }])

    X_input = build_feature_frame(input_data, fset_key)
    if isinstance(X_input, tuple):
        X_input = X_input[0]
    prob = float(model.predict_proba(X_input)[0, 1])
    is_cheat = prob >= threshold

    # Tentukan Kategori Risiko
    if prob >= 0.80:
        risk_level = "KRITIS"
        risk_color = "#DC2626"
    elif prob >= threshold:
        risk_level = "TINGGI"
        risk_color = "#EA580C"
    elif prob >= 0.30:
        risk_level = "SEDANG"
        risk_color = "#D97706"
    else:
        risk_level = "RENDAH (AMAN)"
        risk_color = "#16A34A"

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.markdown(f"### Vonis: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
        st.metric(
            label=f"Probabilitas Risiko Kecurangan ({model_choice.upper()})",
            value=f"{prob*100:.1f}%",
            delta=f"{(prob - threshold)*100:+.1f}% vs Ambang Batas ({threshold*100:.0f}%)",
            delta_color="inverse"
        )
        st.progress(prob)

        if is_cheat:
            st.error(f"🚨 **TERINDIKASI CURANG** (Skor $\\ge$ {threshold*100:.0f}%)")
        else:
            st.success(f"✅ **DIKATEGORIKAN JUJUR** (Skor < {threshold*100:.0f}%)")

    with res_col2:
        st.markdown("#### 📋 Bukti Forensik & Rekomendasi Tindakan:")
        indicators = []
        if (score >= 90) and (duration <= 25):
            indicators.append(r"⚠️ **Speedrun Anomaly:** Skor sangat tinggi ($\ge 90$) dicapai dalam waktu kilat ($\le 25$ menit).")
        if (ip_changes > 0) and (tab_switches >= 3):
            indicators.append("🚨 **Joki / Remote Desktop Anomaly:** Terdeteksi pergantian IP saat aktif berpindah jendela ujian.")
        if peer_similarity >= 0.65:
            indicators.append(f"⚠️ **Peer Collusion Anomaly:** Kemiripan jawaban dengan peserta lain sangat tinggi ({peer_similarity*100:.0f}%).")
        if copy_pastes >= 3:
            indicators.append(f"⚠️ **Clipboard Abuse:** Frekuensi copy-paste teks mencurigakan ({copy_pastes} kali).")
        if mouse_entropy < 0.40:
            indicators.append("⚠️ **Low Mouse Entropy:** Pola gerak kursor mouse sangat minim atau tidak wajar.")

        if not indicators:
            indicators.append("🟢 **Telemetri Bersih:** Tidak ditemukan anomali perilaku yang mencurigakan.")

        for ind in indicators:
            st.markdown(f"- {ind}")

        st.markdown("**Saran Protokol Pengawas:**")
        if prob >= 0.75:
            st.markdown("👉 *Tandai log ujian untuk audit forensik manual, verifikasi rekaman layar / webcam peserta, dan jadwalkan klarifikasi lisan.*")
        elif prob >= threshold:
            st.markdown("👉 *Tinjau riwayat IP dan log perpindahan tab peserta sebelum mengeluarkan nilai resmi.*")
        else:
            st.markdown("👉 *Peserta memenuhi standar integritas ujian CBT. Hasil ujian dapat divalidasi.*")

# ------------------------------------------
# TAB 2: TRANSPARANSI AI (SHAP)
# ------------------------------------------
with tab2:
    st.subheader("Transparansi Keputusan Model (Explainable AI - SHAP)")
    st.markdown(
        "Pendekatan *Explainable AI* menjamin model **bukan kotak hitam (black-box)**. "
        "Setiap keputusan didasari oleh nilai kontribusi matematis (*SHAP values*) yang transparan dan dapat diaudit secara ilmiah."
    )

    shap_summary_file = Path(OUTPUT_DIR) / "shap_summary_plot.png"
    shap_case_file = Path(OUTPUT_DIR) / "shap_individual_case.png"
    shap_csv_file = Path(OUTPUT_DIR) / "shap_importance.csv"

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        if shap_summary_file.exists():
            st.image(
                str(shap_summary_file),
                caption="SHAP Global Summary: Pengaruh Fitur terhadap Vonis Curang (Merah = Nilai Fitur Tinggi)",
                width="stretch",
            )
        else:
            st.info("Jalankan `python -m src.explainability` untuk membuat visualisasi SHAP.")

    with col_s2:
        if shap_csv_file.exists():
            st.markdown("#### 🏆 Top Indikator Paling Berpengaruh:")
            df_imp = pd.read_csv(shap_csv_file)
            df_imp["mean_abs_shap"] = df_imp["mean_abs_shap"].round(4)
            st.dataframe(df_imp.head(7), width="stretch", hide_index=True)
            st.caption("Nilai *Mean |SHAP|* menunjukkan seberapa besar variabel tersebut menggeser probabilitas keputusan model secara rata-rata.")

    if shap_case_file.exists():
        st.markdown("---")
        st.subheader("Studi Kasus Audit 1 Siswa Tertentu (Local Waterfall Plot)")
        st.image(
            str(shap_case_file),
            caption="Waterfall Plot: Rincian kontribusi setiap variabel dalam menaikkan/menurunkan skor kecurangan satu mahasiswa tertentu.",
            width="stretch",
        )

# ------------------------------------------
# TAB 3: PENILAIAN MASSAL (BATCH CSV)
# ------------------------------------------
with tab3:
    st.subheader("Penilaian Otomatis Data Massal (Batch CSV)")
    st.markdown("Unggah file data telemetri (misalnya `data/test.csv`) untuk melakukan audit otomatis ribuan peserta sekaligus:")

    uploaded_file = st.file_uploader("Pilih file CSV log ujian", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"✓ Berhasil membaca **{len(batch_df)}** baris data peserta ujian.")

        X_batch = build_feature_frame(batch_df, fset_key)
        if isinstance(X_batch, tuple):
            X_batch = X_batch[0]

        batch_probs = model.predict_proba(X_batch)[:, 1]
        batch_df["cheat_risk_score"] = np.round(batch_probs * 100, 2)
        batch_df["is_cheating_pred"] = (batch_probs >= threshold).astype(int)

        cheat_cnt = int((batch_df["is_cheating_pred"] == 1).sum())
        honest_cnt = len(batch_df) - cheat_cnt

        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Total Peserta Diaudit", len(batch_df))
        m_c2.metric("Terdeteksi Jujur (✅)", f"{honest_cnt} ({(honest_cnt/len(batch_df))*100:.1f}%)")
        m_c3.metric("Terindikasi Curang (🚨)", f"{cheat_cnt} ({(cheat_cnt/len(batch_df))*100:.1f}%)")
        m_c4.metric("Rata-rata Skor Risiko", f"{batch_probs.mean()*100:.1f}%")

        st.markdown("---")
        st.dataframe(batch_df.head(50), width="stretch")

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            full_csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Unduh Laporan Audit Lengkap (.csv)",
                data=full_csv,
                file_name="laporan_audit_ujian_lengkap.csv",
                mime="text/csv",
                width="stretch",
            )

        with col_dl2:
            id_col = "student_id" if "student_id" in batch_df.columns else batch_df.columns[0]
            submission_df = pd.DataFrame({
                "student_id": batch_df[id_col],
                "is_cheating": batch_df["is_cheating_pred"],
            })
            sub_csv = submission_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Unduh Format Submission Kaggle (.csv)",
                data=sub_csv,
                file_name="submission.csv",
                mime="text/csv",
                width="stretch",
            )

# ------------------------------------------
# TAB 4: PAPAN SKOR MODEL
# ------------------------------------------
with tab4:
    st.subheader("Papan Skor Turnamen Model (Stratified 5-Fold Cross Validation)")
    st.markdown(
        "Evaluasi komparatif berbagai algoritma Machine Learning dan Feature Set "
        "yang diuji menggunakan validasi silang 5-Fold bersusun (*Stratified K-Fold*) dengan `random_state=42`:"
    )

    results_path = Path(OUTPUT_DIR) / "tournament_results.csv"
    if results_path.exists():
        t_df = pd.read_csv(results_path)
        # Ambil kolom esensial dan rapikan format
        display_cols = [c for c in ["model", "feature_set", "n_features", "roc_auc", "f1_at_0.5", "best_f1", "best_threshold", "precision", "recall"] if c in t_df.columns]
        view_df = t_df[display_cols].copy()
        for col in ["roc_auc", "f1_at_0.5", "best_f1", "best_threshold", "precision", "recall"]:
            if col in view_df.columns:
                view_df[col] = view_df[col].apply(lambda x: f"{x:.4f}")
        st.dataframe(view_df, width="stretch", hide_index=True)
    else:
        st.info("File `tournament_results.csv` tidak ditemukan di direktori output.")

    st.markdown("""
    **💡 Catatan Metodologi:**
    1. **Anti Data Leakage:** Preprocessing dan ekstraksi fitur dihitung terpisah di setiap fold pelatihan tanpa menyentuh data validasi.
    2. **Threshold Tuning:** Nilai ambang batas (*threshold*) dioptimasi khusus untuk memaksimalkan *F1-Score* pada data minoritas (kasus kecurangan).
    3. **Ensemble Stacking:** Kombinasi LightGBM dan XGBoost memberikan generalisasi paling stabil di seluruh skenario pengujian.
    """)
