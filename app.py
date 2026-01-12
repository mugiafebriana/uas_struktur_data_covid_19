import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

st.set_page_config(page_title="Deteksi COVID-19 (Rule Based)", layout="centered")

st.title("🦠 Sistem Deteksi COVID-19")
st.markdown("### Menggunakan *Decision Tree Rule-Based System*")

st.divider()

# ======================
# DATASET (11 DATA)
# ======================
data = {
    'Suhu_Tubuh_Tinggi': [1,1,0,1,0,0,1,0,1,0,1,0],
    'Batuk_Kering': [1,1,1,0,0,0,1,0,1,1,0,0],
    'Gangguan_Pernapasan': [1,0,0,0,1,0,1,0,0,1,1,0],
    'Anosmia': [1,1,0,0,0,0,1,0,1,0,1,0],
    'Nyeri_Otot': [1,1,0,1,0,0,1,0,1,1,0,0],
    'Riwayat_Kontak': [1,1,0,0,1,0,1,0,1,1,0,0],
    'Status_Covid': [
        'Positif','Positif','Negatif','Negatif',
        'Positif','Negatif','Positif','Negatif',
        'Positif','Positif','Negatif','Negatif'
    ]
}

df = pd.DataFrame(data)
X = df.drop('Status_Covid', axis=1)
y = df['Status_Covid']

# ======================
# MODEL
# ======================
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)
model.fit(X, y)

# ======================
# RULE DECISION TREE
# ======================
rule_text = export_text(model, feature_names=list(X.columns))

with st.expander("📜 Lihat Rule Decision Tree"):
    st.code(rule_text)

st.divider()

# ======================
# INPUT USER
# ======================
st.subheader("🧪 Input Gejala Pasien")

def pilih(label):
    return st.radio(label, ["Tidak", "Ya"], horizontal=True)

suhu = pilih("Suhu tubuh tinggi")
batuk = pilih("Batuk kering")
napas = pilih("Gangguan pernapasan")
anosmia = pilih("Hilang penciuman")
nyeri = pilih("Nyeri otot")
kontak = pilih("Riwayat kontak")

if st.button("🔍 Proses Deteksi"):
    input_data = pd.DataFrame([{
        'Suhu_Tubuh_Tinggi': 1 if suhu == "Ya" else 0,
        'Batuk_Kering': 1 if batuk == "Ya" else 0,
        'Gangguan_Pernapasan': 1 if napas == "Ya" else 0,
        'Anosmia': 1 if anosmia == "Ya" else 0,
        'Nyeri_Otot': 1 if nyeri == "Ya" else 0,
        'Riwayat_Kontak': 1 if kontak == "Ya" else 0
    }])

    hasil = model.predict(input_data)[0]

    st.divider()

    st.subheader("📌 Hasil Diagnosa")
    if hasil == "Positif":
        st.error("⚠️ PASIEN TERDETEKSI COVID-19")
    else:
        st.success("✅ PASIEN TIDAK TERDETEKSI COVID-19")

    st.markdown("### 🧠 Keputusan Berdasarkan Rule Decision Tree")
    st.info("Hasil ditentukan berdasarkan rule yang dihasilkan dari Decision Tree, bukan keputusan manual.")
