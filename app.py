import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Deteksi COVID-19", layout="centered")

st.title("🦠 Deteksi COVID-19")
st.write("Aplikasi sederhana deteksi COVID-19 menggunakan Decision Tree")

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
# INPUT USER
# ======================
st.subheader("Masukkan Gejala:")

suhu = st.checkbox("Suhu tubuh tinggi")
batuk = st.checkbox("Batuk kering")
napas = st.checkbox("Gangguan pernapasan")
anosmia = st.checkbox("Hilang penciuman")
nyeri = st.checkbox("Nyeri otot")
kontak = st.checkbox("Riwayat kontak")

if st.button("🔍 Deteksi"):
    data_pasien = pd.DataFrame([{
        'Suhu_Tubuh_Tinggi': int(suhu),
        'Batuk_Kering': int(batuk),
        'Gangguan_Pernapasan': int(napas),
        'Anosmia': int(anosmia),
        'Nyeri_Otot': int(nyeri),
        'Riwayat_Kontak': int(kontak)
    }])

    hasil = model.predict(data_pasien)[0]

    if hasil == "Positif":
        st.error("⚠️ TERDETEKSI COVID-19")
    else:
        st.success("✅ TIDAK TERDETEKSI COVID-19")
