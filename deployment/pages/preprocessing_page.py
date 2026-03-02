import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from utils.data_loader import load_data

def show():
    st.header("Preprocessing")

    data = load_data()

    if data is None:
        return

    # Tahap 1: Imputasi dengan KNN
    st.subheader("1. Imputasi dengan KNN")
    st.markdown("""
    Tahap pertama adalah mengimputasi missing values (nilai 0) pada kolom tertentu menggunakan KNN Imputer.
    """)

    kolom_imputasi = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    st.write("**Kolom yang diimputasi:**", kolom_imputasi)

    data_copy = data.copy()

    for col in kolom_imputasi:
        data_copy.loc[data_copy[col] == 0, col] = np.nan

    st.write("**Jumlah missing values setelah konversi nilai 0 ke NaN:**")
    missing_info = data_copy[kolom_imputasi].isnull().sum()
    st.dataframe(pd.DataFrame({'Fitur': missing_info.index, 'Missing Values': missing_info.values}))

    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(data_copy.drop('Outcome', axis=1))
    data_imputed = pd.DataFrame(X_imputed, columns=data_copy.drop('Outcome', axis=1).columns)
    data_imputed['Outcome'] = data_copy['Outcome'].values

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sample Data Sebelum Imputasi:**")
        st.dataframe(data.head(5))

    with col2:
        st.markdown("**Sample Data Setelah Imputasi:**")
        st.dataframe(data_imputed.head(5).round(2))

    # Tahap 2: Normalisasi
    st.subheader("2. Normalisasi dengan MinMax Scaler")
    st.markdown("""
    Setelah imputasi, seluruh data dinormalisasi menggunakan MinMax Scaler untuk mengubah rentang nilai
    menjadi 0-1 agar tidak ada fitur yang mendominasi karena perbedaan skala.
    """)

    scaler = MinMaxScaler()
    X_for_scaling = data_imputed.drop('Outcome', axis=1)
    X_scaled = pd.DataFrame(scaler.fit_transform(X_for_scaling),
                           columns=X_for_scaling.columns,
                           index=X_for_scaling.index)
    data_normalized = X_scaled.copy()
    data_normalized['Outcome'] = data_imputed['Outcome'].values

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sample Data Sebelum Normalisasi:**")
        st.dataframe(data_imputed.head(5).round(2))

    with col2:
        st.markdown("**Sample Data Setelah Normalisasi:**")
        st.dataframe(data_normalized.head(5).round(4))

    # Tahap 3: Split Data
    st.subheader("3. Split Data")
    st.markdown("**Data yang sudah dipreprocessing dibagi dengan rasio 70:30**")

    X = data_normalized.drop('Outcome', axis=1)
    y = data_normalized['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Data Training (70%)", len(X_train))
        st.markdown("**Distribusi Training:**")

        fig, ax = plt.subplots(figsize=(6, 4))
        train_counts = y_train.value_counts()
        bars = ax.bar(['Non Diabetes', 'Diabetes'], train_counts.values,
                     color=['skyblue', 'orange'], alpha=0.7)
        ax.set_ylabel('Jumlah')
        ax.set_title('Distribusi Data Training')

        for i, v in enumerate(train_counts.values):
            ax.text(i, v + 5, str(v), ha='center')
        st.pyplot(fig)

    with col2:
        st.metric("Data Testing (30%)", len(X_test))
        st.markdown("**Distribusi Testing:**")

        fig, ax = plt.subplots(figsize=(6, 4))
        test_counts = y_test.value_counts()
        bars = ax.bar(['Non Diabetes', 'Diabetes'], test_counts.values,
                     color=['lightcoral', 'gold'], alpha=0.7)
        ax.set_ylabel('Jumlah')
        ax.set_title('Distribusi Data Testing')

        for i, v in enumerate(test_counts.values):
            ax.text(i, v + 5, str(v), ha='center')
        st.pyplot(fig)

    # Tahap 4: SMOTE
    st.subheader("4. SMOTE (Balancing)")
    st.markdown("""
    Pada tahap terakhir dilakukan penyeimbangan data training menggunakan **SMOTE (Synthetic Minority Oversampling Technique)**
    untuk mengatasi ketidakseimbangan kelas.
    """)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sebelum SMOTE:**")
        before_counts = [sum(y_train == 0), sum(y_train == 1)]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(['Non Diabetes', 'Diabetes'], before_counts,
                     color=['skyblue', 'orange'], alpha=0.7)
        ax.set_ylabel('Jumlah')
        ax.set_title('Distribusi Sebelum SMOTE')

        for i, v in enumerate(before_counts):
            ax.text(i, v + 5, str(v), ha='center')
        st.pyplot(fig)

        before_smote = pd.DataFrame({
            'Class': ['Non Diabetes', 'Diabetes'],
            'Count': before_counts
        })
        st.dataframe(before_smote)

    with col2:
        st.markdown("**Setelah SMOTE:**")
        after_counts = [sum(y_train_smote == 0), sum(y_train_smote == 1)]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(['Non Diabetes', 'Diabetes'], after_counts,
                     color=['lightgreen', 'lightcoral'], alpha=0.7)
        ax.set_ylabel('Jumlah')
        ax.set_title('Distribusi Setelah SMOTE')

        for i, v in enumerate(after_counts):
            ax.text(i, v + 5, str(v), ha='center')
        st.pyplot(fig)

        after_smote = pd.DataFrame({
            'Class': ['Non Diabetes', 'Diabetes'],
            'Count': after_counts
        })
        st.dataframe(after_smote)

    # Summary
    st.subheader("5. Data Akhir untuk Modelling")
    original_diabetes = sum(y_train == 1)
    synthetic_added = after_counts[1] - original_diabetes

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data Training Final", f"{len(X_train_smote):,}")
    with col2:
        st.metric("Data Sintetis Ditambahkan", f"{synthetic_added:,}")
    with col3:
        st.metric("Balance Ratio", "50:50")