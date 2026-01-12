import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diagnosa COVID-19", layout="wide")

st.title("🦠 Diagnosa COVID-19 - Decision Tree")
st.write("UAS Struktur Data")

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

st.subheader("📊 Tabel Data Sample (11 Data)")
st.dataframe(df)

# ======================
# MODEL DECISION TREE
# ======================
X = df.drop('Status_Covid', axis=1)
y = df['Status_Covid']

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)
model.fit(X, y)

# ======================
# VISUALISASI TREE
# ======================
st.subheader("🌳 Diagram Decision Tree")

fig, ax = plt.subplots(figsize=(22, 12))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,      # WARNA KUNING AKTIF
    rounded=True,
    ax=ax
)
st.pyplot(fig)

st.caption("Node berwarna menunjukkan rule (sesuai permintaan dosen)")

# ======================
# INPUT USER
# ======================
st.subheader("🧪 Input Gejala Pasien")

col1, col2, col3 = st.columns(3)

with col1:
    suhu = st.checkbox("Suhu tubuh tinggi")
    batuk = st.checkbox("Batuk kering")

with col2:
    napas = st.checkbox("Gangguan pernapasan")
    anosmia = st.checkbox("Hilang penciuman")

with col3:
    nyeri = st.checkbox("Nyeri otot")
    kontak = st.checkbox("Riwayat kontak")

if st.button("🔍 Prediksi COVID-19"):
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
        st.error("⚠️ Pasien TERINDIKASI COVID-19")
    else:
        st.success("✅ Pasien TIDAK TERINDIKASI COVID-19")
