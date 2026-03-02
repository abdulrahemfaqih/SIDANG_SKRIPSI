import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_loader import load_data

def show():
    st.header("EDA (Exploratory Data Analysis)")
    st.markdown("""
Pada halaman ini, terdapat penjelasan mengenai dataset yang digunakan dan distribusi dari setiap fitur.

**Sumber Dataset:**
Dataset yang digunakan adalah data sekunder yang berasal dari Hospital Frankfurt, Germany, sebagaimana yang telah dilakukan oleh penelitian sebelumnya oleh Asfandyar Khan et al. pada link berikut https://www.mdpi.com/2075-4418/12/11/2595
""")

    data = load_data()

    if data is not None:
        st.subheader("Informasi Dasar Dataset")

        # Dataset info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Jumlah Baris", data.shape[0])
            st.metric("Jumlah Fitur", data.shape[1])

        with col2:
            st.metric("Missing Values", data.isnull().sum().sum())
            st.metric("Target Classes", data['Outcome'].nunique())

        # Penjelasan Fitur
        st.subheader("Penjelasan Fitur")
        st.markdown("""
Dataset ini memiliki 9 fitur dengan penjelasan sebagai berikut:

1. **Pregnancies**: Jumlah kehamilan yang pernah dialami pasien
2. **Glucose**: Konsentrasi glukosa plasma dalam tes toleransi glukosa oral 2 jam (mg/dL)
3. **BloodPressure**: Tekanan darah diastolik (mm Hg)
4. **SkinThickness**: Ketebalan lipatan kulit trisep (mm)
5. **Insulin**: Insulin serum 2 jam (mu U/ml)
6. **BMI**: Body Mass Index atau Indeks Massa Tubuh (berat dalam kg/(tinggi dalam m)^2)
7. **DiabetesPedigreeFunction**: Fungsi silsilah diabetes (skor yang menunjukkan riwayat diabetes dalam keluarga)
8. **Age**: Usia pasien (tahun)
9. **Outcome**: Variabel target (0 = Non-Diabetes, 1 = Diabetes)
""")

        # Tampilkan dataset
        st.subheader("Dataset Sample")
        st.dataframe(data.head(10))

        # Missing values analysis
        st.subheader("Analisis Nilai 0 (Missing Values)")
        st.markdown("""Pada dataset ini, nilai 0 pada kolom tertentu menunjukkan missing values yang perlu diimputasi:""")

        zero_counts = {}
        kolom_imputasi = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in data.columns:
            if col != 'Outcome' and col in kolom_imputasi:
                zero_counts[col] = (data[col] == 0).sum()

        zero_df = pd.DataFrame(list(zero_counts.items()), columns=['Fitur', 'Jumlah Nilai 0'])
        st.dataframe(zero_df)

        total_zeros = sum(zero_counts.values())
        fitur_dengan_nol = data[kolom_imputasi] == 0
        total_baris_nol = fitur_dengan_nol.any(axis=1).sum()
        persentase_baris_nol = (total_baris_nol / len(data)) * 100

        st.info(f"""
        - Total nilai 0 yang akan diimputasi: **{total_zeros}**
        - Jumlah baris yang memiliki setidaknya satu nilai 0: **{total_baris_nol}** dari **{len(data)}**
        - Persentase baris dengan missing value: **{persentase_baris_nol:.2f}%**
        """)

        # Distribusi target
        st.subheader("Distribusi Target Class")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            outcome_counts = data['Outcome'].value_counts()
            bars = ax.bar(['Non Diabetes', 'Diabetes'], outcome_counts.values,
                         color=['skyblue', 'orange'])
            ax.set_ylabel('Jumlah')
            ax.set_title('Distribusi Kelas')

            for i, v in enumerate(outcome_counts.values):
                ax.text(i, v + 5, str(v), ha='center')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(outcome_counts.values, labels=['Non Diabetes', 'Diabetes'],
                   autopct='%1.1f%%', colors=['skyblue', 'orange'])
            ax.set_title('Proporsi Kelas')
            st.pyplot(fig)

        # Distribusi fitur
        st.subheader("Distribusi Fitur")
        feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        selected_feature = st.selectbox("Pilih Fitur untuk Visualisasi:", feature_cols)

        fig, ax = plt.subplots(figsize=(10, 6))
        for outcome in [0, 1]:
            subset = data[data['Outcome'] == outcome][selected_feature]
            label = 'Non Diabetes' if outcome == 0 else 'Diabetes'
            ax.hist(subset, alpha=0.7, label=label, bins=20)
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Frekuensi')
        ax.legend()
        ax.set_title(f'Distribusi {selected_feature}')
        st.pyplot(fig)