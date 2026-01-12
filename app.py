import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# =========================
# JUDUL
# =========================
st.title("Deteksi COVID-19 Menggunakan Decision Tree")
st.write("Aplikasi berbasis aturan (rule-based)")

# =========================
# DATASET (DARI COLAB KAMU)
# =========================
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

# =========================
# MODEL
# =========================
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=42
)
model.fit(X, y)

# =========================
# INPUT USER (STREAMLIT)
# =========================
st.subheader("Input Gejala Pasien")

def pilih(label):
    return st.selectbox(label, ["Tidak", "Ya"])

input_data = {
    'Suhu_Tubuh_Tinggi': pilih("Suhu Tubuh Tinggi"),
    'Batuk_Kering': pilih("Batuk Kering"),
    'Gangguan_Pernapasan': pilih("Gangguan Pernapasan"),
    'Anosmia': pilih("Hilang Penciuman (Anosmia)"),
    'Nyeri_Otot': pilih("Nyeri Otot"),
    'Riwayat_Kontak': pilih("Riwayat Kontak")
}

def konversi(val):
    return 1 if val == "Ya" else 0

df_pasien = pd.DataFrame([{
    k: konversi(v) for k, v in input_data.items()
}])

# =========================
# PREDIKSI
# =========================
if st.button("Prediksi"):
    hasil = model.predict(df_pasien)[0]

    if hasil == "Positif":
        st.error("⚠️ HASIL: POSITIF COVID-19")
    else:
        st.success("✅ HASIL: NEGATIF COVID-19")
