import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="PENERAPAN METODE STACKING ENSEMBLE PADA KLASIFIKASI PENYAKIT DIABETES MELITUS",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('data/diabetes_frankurt_germany.csv')
        return data
    except:
        st.error("File diabetes_frankurt_germany.csv tidak ditemukan!")
        return None

# Fungsi untuk load model
@st.cache_resource
def load_models():
    try:
        imputer = joblib.load('model_deployment/knn_imputer.pkl')
        scaler = joblib.load('model_deployment/minmax_scaler.pkl')

        # Load base models
        lr_model = joblib.load('model_deployment/logistic_regression.pkl')
        knn_model = joblib.load('model_deployment/knn.pkl')
        svm_model = joblib.load('model_deployment/svm.pkl')
        mlp_model = joblib.load('model_deployment/mlp.pkl')

        # Load meta learner
        meta_model = joblib.load('model_deployment/mlp_meta.pkl')

        # Load info
        info = joblib.load('model_deployment/model_info.pkl')

        return imputer, scaler, lr_model, knn_model, svm_model, mlp_model, meta_model, info
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Fungsi prediksi
def predict_diabetes(data_input, models):
    imputer, scaler, lr_model, knn_model, svm_model, mlp_model, meta_model, info = models

    # Preprocessing
    df = pd.DataFrame([data_input])

    # Imputasi
    for col in info['kolom_imputasi']:
        if df[col].iloc[0] == 0:
            df[col] = np.nan

    df_imputed = pd.DataFrame(imputer.transform(df), columns=info['fitur'])
    df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=info['fitur'])

    # Prediksi base models
    lr_prob = lr_model.predict_proba(df_scaled)
    knn_prob = knn_model.predict_proba(df_scaled)
    svm_prob = svm_model.predict_proba(df_scaled)
    mlp_prob = mlp_model.predict_proba(df_scaled)

    # Gabungkan meta features
    meta_features = np.column_stack([lr_prob, knn_prob, svm_prob, mlp_prob])

    # Prediksi final
    result = meta_model.predict(meta_features)[0]
    prob = meta_model.predict_proba(meta_features)[0]

    # Prepare meta features untuk display
    meta_df = pd.DataFrame(meta_features, columns=[
        'LR_prob_0', 'LR_prob_1', 'KNN_prob_0', 'KNN_prob_1',
        'SVM_prob_0', 'SVM_prob_1', 'MLP_prob_0', 'MLP_prob_1'
    ])

    return {
        'prediksi': 'Diabetes' if result == 1 else 'Non Diabetes',
        'probabilitas_diabetes': prob[1],
        'probabilitas_non_diabetes': prob[0],
        'confidence': max(prob),
        'meta_features': meta_df,
        'data_preprocessed': df_scaled
    }

# Title
st.title("PENERAPAN METODE STACKING ENSEMBLE PADA KLASIFIKASI PENYAKIT DIABETES MELITUS")

# Sidebar
st.sidebar.title("Menu")
tabs = ["Data", "Preprocessing", "Modelling", "Testing"]
selected_tab = st.sidebar.selectbox("Pilih Halaman:", tabs)

# Load data
data = load_data()

if data is not None:
    if selected_tab == "Data":
        st.header("EDA (Exploratory Data Analysis)")
        st.markdown("""
Pada halaman ini, terdapat penjelasan mengenai dataset yang digunakan dan distribusi dari setiap fitur.

**Sumber Dataset:**
Dataset yang digunakan adalah data sekunder yang berasal dari Hospital Frankfurt, Germany, sebagaimana yang telah dilakukan oleh penelitian sebelumnya oleh Asfandyar Khan et al. pada link berikut https://www.mdpi.com/2075-4418/12/11/2595
""")
        st.subheader("Informasi Dasar Dataset")

        # Dataset info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Jumlah Baris", data.shape[0])
            st.metric("Jumlah Fitur", data.shape[1])

        with col2:
            st.metric("Missing Values", data.isnull().sum().sum())
            st.metric("Target Classes", data['Outcome'].nunique())

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

        zero_df = pd.DataFrame(list(zero_counts.items()),
                              columns=['Fitur', 'Jumlah Nilai 0'])
        st.dataframe(zero_df)

         # Hitung total dan persentase
        total_zeros = sum(zero_counts.values())
        fitur_dengan_nol = data[kolom_imputasi] == 0
        total_baris_nol = fitur_dengan_nol.any(axis=1).sum()
        persentase_baris_nol = (total_baris_nol / len(data)) * 100


        st.info(f"""
        - Total nilai 0 yang akan diimputasi: **{total_zeros}**
        - Jumlah baris yang memiliki setidaknya satu nilai 0 pada kolom imputasi: **{total_baris_nol}** dari **{len(data)}** baris
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

    elif selected_tab == "Preprocessing":
        st.header("Preprocessing")

        # Tahap 1: Imputasi dengan KNN
        st.subheader("1. Imputasi dengan KNN")
        st.markdown("""
        Tahap pertama adalah mengimputasi missing values (nilai 0) pada kolom tertentu menggunakan KNN Imputer.
        """)

        kolom_imputasi = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        st.write("**Kolom yang diimputasi:**", kolom_imputasi)

        # Proses imputasi pada data asli
        data_copy = data.copy()

        # Konversi nilai 0 ke NaN untuk kolom imputasi
        for col in kolom_imputasi:
            data_copy.loc[data_copy[col] == 0, col] = np.nan

        st.write("**Jumlah missing values setelah konversi nilai 0 ke NaN:**")
        missing_info = data_copy[kolom_imputasi].isnull().sum()
        st.dataframe(pd.DataFrame({'Fitur': missing_info.index,
                                  'Missing Values': missing_info.values}))

        # Lakukan imputasi KNN pada seluruh data
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = imputer.fit_transform(data_copy.drop('Outcome', axis=1))
        data_imputed = pd.DataFrame(X_imputed, columns=data_copy.drop('Outcome', axis=1).columns)
        data_imputed['Outcome'] = data_copy['Outcome'].values

        # Tampilkan sample data sebelum dan sesudah imputasi
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sample Data Sebelum Imputasi:**")
            sample_before = data.head(5)
            st.dataframe(sample_before)

        with col2:
            st.markdown("**Sample Data Setelah Imputasi:**")
            sample_after_impute = data_imputed.head(5)
            st.dataframe(sample_after_impute.round(2))

        # Tahap 2: Normalisasi
        st.subheader("2. Normalisasi dengan MinMax Scaler")
        st.markdown("""
        Setelah imputasi, seluruh data dinormalisasi menggunakan MinMax Scaler untuk mengubah rentang nilai
        menjadi 0-1 agar tidak ada fitur yang mendominasi karena perbedaan skala.
        """)

        # Lakukan normalisasi pada seluruh data
        scaler = MinMaxScaler()
        X_for_scaling = data_imputed.drop('Outcome', axis=1)
        X_scaled = pd.DataFrame(scaler.fit_transform(X_for_scaling),
                               columns=X_for_scaling.columns,
                               index=X_for_scaling.index)
        data_normalized = X_scaled.copy()
        data_normalized['Outcome'] = data_imputed['Outcome'].values

        # Tampilkan sample data sebelum dan sesudah normalisasi
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sample Data Sebelum Normalisasi:**")
            st.dataframe(sample_after_impute.round(2))

        with col2:
            st.markdown("**Sample Data Setelah Normalisasi:**")
            sample_normalized = data_normalized.head(5)
            st.dataframe(sample_normalized.round(4))

        # Tahap 3: Split Data
        st.subheader("3. Split Data")
        st.markdown("**Data yang sudah dipreprocessing dibagi dengan rasio 70:30**")

        X = data_normalized.drop('Outcome', axis=1)
        y = data_normalized['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Data Training (70%)", len(X_train))
            st.markdown("**Distribusi Training:**")

            # Bar chart untuk training data
            fig, ax = plt.subplots(figsize=(6, 4))
            train_counts = y_train.value_counts()
            bars = ax.bar(['Non Diabetes', 'Diabetes'], train_counts.values,
                         color=['skyblue', 'orange'], alpha=0.7)
            ax.set_ylabel('Jumlah')
            ax.set_title('Distribusi Data Training')

            # Tambah nilai di atas bar
            for i, v in enumerate(train_counts.values):
                ax.text(i, v + 5, str(v), ha='center')

            st.pyplot(fig)

        with col2:
            st.metric("Data Testing (30%)", len(X_test))
            st.markdown("**Distribusi Testing:**")

            # Bar chart untuk testing data
            fig, ax = plt.subplots(figsize=(6, 4))
            test_counts = y_test.value_counts()
            bars = ax.bar(['Non Diabetes', 'Diabetes'], test_counts.values,
                         color=['lightcoral', 'gold'], alpha=0.7)
            ax.set_ylabel('Jumlah')
            ax.set_title('Distribusi Data Testing')

            # Tambah nilai di atas bar
            for i, v in enumerate(test_counts.values):
                ax.text(i, v + 5, str(v), ha='center')

            st.pyplot(fig)

        # Tahap 4: SMOTE (Balancing)
        st.subheader("4. SMOTE (Balancing)")
        st.markdown("""
        Pada tahap terakhir dilakukan penyeimbangan data training menggunakan **SMOTE (Synthetic Minority Oversampling Technique)**
        untuk mengatasi ketidakseimbangan kelas, khususnya pada kelas Diabetes yang merupakan minority class.
        """)

        # Lakukan SMOTE pada data training
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sebelum SMOTE:**")

            # Data sebelum SMOTE
            before_counts = [sum(y_train == 0), sum(y_train == 1)]

            # Bar chart sebelum SMOTE
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(['Non Diabetes', 'Diabetes'], before_counts,
                         color=['skyblue', 'orange'], alpha=0.7)
            ax.set_ylabel('Jumlah')
            ax.set_title('Distribusi Sebelum SMOTE')

            # Tambah nilai di atas bar
            for i, v in enumerate(before_counts):
                ax.text(i, v + 5, str(v), ha='center')

            st.pyplot(fig)

            # Tabel sebelum SMOTE
            before_smote = pd.DataFrame({
                'Class': ['Non Diabetes', 'Diabetes'],
                'Count': before_counts
            })
            st.dataframe(before_smote)

        with col2:
            st.markdown("**Setelah SMOTE:**")

            # Data setelah SMOTE
            after_counts = [sum(y_train_smote == 0), sum(y_train_smote == 1)]

            # Bar chart setelah SMOTE
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(['Non Diabetes', 'Diabetes'], after_counts,
                         color=['lightgreen', 'lightcoral'], alpha=0.7)
            ax.set_ylabel('Jumlah')
            ax.set_title('Distribusi Setelah SMOTE')

            # Tambah nilai di atas bar
            for i, v in enumerate(after_counts):
                ax.text(i, v + 5, str(v), ha='center')

            st.pyplot(fig)

            # Tabel setelah SMOTE
            after_smote = pd.DataFrame({
                'Class': ['Non Diabetes', 'Diabetes'],
                'Count': after_counts
            })
            st.dataframe(after_smote)

        # Summary final preprocessing
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


    elif selected_tab == "Modelling":
        st.header("Training Model")

        st.subheader("Arsitektur Stacking Ensemble")
        st.markdown("""
        **Base Models (Level 1):**
        - Logistic Regression
        - K-Nearest Neighbors (KNN)
        - Support Vector Machine (SVM)
        - Multi-Layer Perceptron (MLP)

        **Meta Learner (Level 2):**
        Setiap base model akan dicoba secara bergantian sebagai meta learner untuk menentukan yang terbaik:
        - **Skenario 1**: KNN sebagai meta learner
        - **Skenario 2**: SVM sebagai meta learner
        - **Skenario 3**: MLP sebagai meta learner
        - **Skenario 4**: Logistic Regression sebagai meta learner
        """)


        # Cross validation explanation
        st.subheader("Cross Validation Setup")
        st.markdown("""
        **K-Fold Cross Validation dengan 5 fold**  pada data yang sudah dipreprocessing lengkap (imputasi + normalisasi + SMOTE):
        - Dataset yang sudah di-SMOTE dibagi menjadi 5 fold secara acak
        - Model dilatih pada 4 fold dan ditest pada 1 fold
        - Proses diulang 5 kali dengan fold yang berbeda sebagai validation set
        """)

        # Simulasi CV splits menggunakan data setelah preprocessing lengkap
        from sklearn.model_selection import KFold

        # Gunakan data hasil preprocessing lengkap (setelah SMOTE)
         # Load data dari CSV yang sudah disimpan
        @st.cache_data
        def load_preprocessed_data():
            try:
                # Load data training setelah SMOTE
                train_data = pd.read_csv('data/data_training_smote.csv')
                X_final = train_data.drop('Outcome', axis=1)
                y_final = train_data['Outcome']

                # Load data testing
                test_data = pd.read_csv('data/data_testing_normalized.csv')
                X_test_final = test_data.drop('Outcome', axis=1)
                y_test_final = test_data['Outcome']

                return X_final, y_final, X_test_final, y_test_final
            except FileNotFoundError:
                st.error("File CSV preprocessing tidak ditemukan! Jalankan preprocessing terlebih dahulu.")
                return None, None, None, None
        X_final, y_final, X_test_final, y_test_final = load_preprocessed_data()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        cv_info = []
        fold_num = 1

        for train_idx, test_idx in kf.split(X_final):
            train_size = len(train_idx)
            test_size = len(test_idx)

            # Hitung distribusi kelas di train dan test
            train_class_0 = sum(y_final.iloc[train_idx] == 0)
            train_class_1 = sum(y_final.iloc[train_idx] == 1)
            test_class_0 = sum(y_final.iloc[test_idx] == 0)
            test_class_1 = sum(y_final.iloc[test_idx] == 1)

            cv_info.append({
                'Fold': fold_num,
                'Train_Size': train_size,
                'Test_Size': test_size,
                'Train_Non_Diabetes': train_class_0,
                'Train_Diabetes': train_class_1,
                'Test_Non_Diabetes': test_class_0,
                'Test_Diabetes': test_class_1,
                'Train_Diabetes_Ratio': f"{(train_class_1/train_size)*100:.1f}%",
                'Test_Diabetes_Ratio': f"{(test_class_1/test_size)*100:.1f}%"
            })

            fold_num += 1

        # Tampilkan tabel CV splits
        st.subheader("Detail Cross Validation Splits")
        st.markdown("*Data yang digunakan: Hasil preprocessing lengkap (Imputasi + Normalisasi + SMOTE)*")
        cv_df = pd.DataFrame(cv_info)
         # Buat tabel seperti gambar
        total_samples = len(X_final)
        samples_per_fold = total_samples // 5
        remaining = total_samples % 5

        # Hitung ukuran setiap fold
        fold_sizes = []
        for i in range(5):
            if i < remaining:
                fold_sizes.append(samples_per_fold + 1)
            else:
                fold_sizes.append(samples_per_fold)

        # Buat data untuk tabel
        cv_matrix_data = []
        for iterasi in range(1, 6):
            row = {'Iterasi': f'Iterasi {iterasi}'}
            for fold in range(1, 6):
                if fold == iterasi:  # Fold ini jadi validation set
                    row[f'Fold {fold}'] = f"{fold_sizes[fold-1]} (Val)"
                else:  # Fold ini jadi training set
                    row[f'Fold {fold}'] = f"{fold_sizes[fold-1]} (Train)"
            cv_matrix_data.append(row)

        cv_matrix_df = pd.DataFrame(cv_matrix_data)

        # Function untuk styling
        def highlight_validation(val):
            if '(Val)' in str(val):
                return 'background-color: #ffeb3b; font-weight: bold'  # Kuning untuk validation
            elif '(Train)' in str(val):
                return 'background-color: #e3f2fd'  # Biru muda untuk training
            return ''

        # Tampilkan tabel dengan styling
        st.dataframe(
            cv_matrix_df.style.applymap(highlight_validation).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]},
                {'selector': 'td', 'props': [('text-align', 'center')]},
                {'selector': '.col0', 'props': [('background-color', '#f8f9fa'), ('font-weight', 'bold')]}
            ]),
            use_container_width=True,
            hide_index=True
        )

        # Keterangan warna
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("🟨 **Validation Set** - Data untuk evaluasi model")
        with col2:
            st.markdown("🟦 **Training Set** - Data untuk melatih model")


        # Visualisasi CV splits
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Ukuran Data per Iterasi:**")
            fig, ax = plt.subplots(figsize=(8, 5))

            x = np.arange(len(cv_df))
            width = 0.35

            bars1 = ax.bar(x - width/2, cv_df['Train_Size'], width,
                          label='Training', color='skyblue', alpha=0.8)
            bars2 = ax.bar(x + width/2, cv_df['Test_Size'], width,
                          label='Validation', color='orange', alpha=0.8)

            ax.set_xlabel('Iterasi Cross Validation')
            ax.set_ylabel('Jumlah Sampel')
            ax.set_title('Pembagian Data Training-Validation per Iterasi CV\n(Data Setelah SMOTE)')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Iterasi {i+1}' for i in range(5)])
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Tambah nilai di atas bar
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)

            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
            st.pyplot(fig)
        with col2:
            st.markdown("**Distribusi Kelas per Iterasi CV:**")
            fig, ax = plt.subplots(figsize=(8, 5))

            # Stacked bar chart untuk distribusi kelas
            train_non_diabetes = cv_df['Train_Non_Diabetes'].values
            train_diabetes = cv_df['Train_Diabetes'].values

            x = np.arange(len(cv_df))

            bars1 = ax.bar(x, train_non_diabetes, label='Non Diabetes',
                          color='lightblue', alpha=0.8)
            bars2 = ax.bar(x, train_diabetes, bottom=train_non_diabetes,
                          label='Diabetes', color='lightcoral', alpha=0.8)

            ax.set_xlabel('Iterasi Cross Validation')
            ax.set_ylabel('Jumlah Sampel Training')
            ax.set_title('Distribusi Kelas dalam Data Training per Iterasi CV\n(Data Setelah SMOTE)')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Iterasi {i+1}' for i in range(5)])
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)


                # Parameter grids
        st.subheader("Hyperparameter Tuning")
        st.markdown("""
        Setiap model menggunakan **GridSearchCV** dengan 5-fold cross validation untuk mencari parameter optimal.
        Berikut adalah grid parameter yang digunakan untuk setiap model:
        """)

        param_info = {
            'Logistic Regression': {
                'C': 'np.logspace(-4, 4, 20)',
                'solver': "['liblinear', 'lbfgs']",
                'penalty': "['l1', 'l2']"
            },
            'KNN': {
                'n_neighbors': '[5, 7, 9, 11, 13, 15, 17, 19]',
                'weights': "['uniform', 'distance']",
                'metric': "['euclidean', 'manhattan', 'minkowski']"
            },
            'SVM': {
                'C': '[0.1, 1, 10, 100, 1000]',
                'gamma': "['scale', 'auto']",
                'kernel': "['rbf', 'linear']"
            },
            'MLP': {
                'max_iter': '[100, 500, 1000]',
                'hidden_layer_sizes': '[(100,), (50,50), (30,20), (20,10)]',
                'activation': "['relu', 'tanh', 'logistic']",
                'alpha': '[0.0001, 0.001, 0.01, 0.1]'
            }
        }

        # Tampilkan semua parameter dalam layout yang rapi
        col1, col2 = st.columns(2)

        with col1:
            # Logistic Regression
            st.markdown("### Logistic Regression")
            for param, values in param_info['Logistic Regression'].items():
                st.markdown(f"**{param}**: `{values}`")

            # KNN
            st.markdown("### K-Nearest Neighbors (KNN)")
            for param, values in param_info['KNN'].items():
                st.markdown(f"**{param}**: `{values}`")

        with col2:
            # SVM
            st.markdown("### Support Vector Machine (SVM)")
            for param, values in param_info['SVM'].items():
                st.markdown(f"**{param}**: `{values}`")

            # MLP
            st.markdown("### Multi-Layer Perceptron (MLP)")
            for param, values in param_info['MLP'].items():
                st.markdown(f"**{param}**: `{values}`")


                # Hasil Training Model
        st.subheader("Hasil Training Model")
        st.markdown("""
        Berikut adalah hasil hyperparameter tuning dan cross validation untuk setiap model menggunakan **metrik akurasi**:
        """)

        # Data hasil training BASE MODELS
        base_models_results = {
            'Model': ['Logistic Regression', 'KNN', 'SVM', 'MLP'],
            'Best CV Score': [0.7481, 0.9445, 0.9180, 0.8290],
            'Fold 1': [0.7601, 0.9461, 0.9326, 0.8113],
            'Fold 2': [0.7089, 0.9380, 0.9218, 0.8329],
            'Fold 3': [0.7466, 0.9326, 0.9057, 0.8329],
            'Fold 4': [0.7655, 0.9434, 0.9030, 0.8194],
            'Fold 5': [0.7595, 0.9622, 0.9270, 0.8486],
            'Std Dev': [0.0206, 0.0100, 0.0117, 0.0128],
            'Best Parameters': [
                "C: 0.0127, penalty: 'l2', solver: 'liblinear'",
                "n_neighbors: 5, weights: 'distance', metric: 'manhattan'",
                "C: 1000, gamma: 'scale', kernel: 'rbf'",
                "hidden_layers: (50,50), activation: 'relu', alpha: 0.01"
            ]
        }

        # Data hasil training META LEARNERS
        meta_learners_results = {
            'Meta Learner': ['KNN Meta', 'SVM Meta', 'MLP Meta', 'LogReg Meta'],
            'Best CV Score': [0.9455, 0.9450, 0.9472, 0.9439],
            'Fold 1': [0.9542, 0.9515, 0.9596, 0.9542],
            'Fold 2': [0.9488, 0.9434, 0.9434, 0.9380],
            'Fold 3': [0.9326, 0.9353, 0.9407, 0.9380],
            'Fold 4': [0.9326, 0.9380, 0.9272, 0.9326],
            'Fold 5': [0.9595, 0.9568, 0.9649, 0.9568],
            'Std Dev': [0.0111, 0.0081, 0.0136, 0.0097],
            'Best Parameters': [
                "n_neighbors: 19, weights: 'uniform', metric: 'manhattan'",
                "C: 1000, gamma: 'scale', kernel: 'linear'",
                "hidden_layers: (20,10), activation: 'logistic', alpha: 0.01",
                "C: 0.616, penalty: 'l1', solver: 'liblinear'"
            ]
        }

        # Tampilkan tabel BASE MODELS
        st.markdown("### Base Models Performance")
        base_df = pd.DataFrame(base_models_results)

        st.dataframe(
            base_df.style.format({
                'Best CV Score': '{:.4f}',
                'Fold 1': '{:.4f}',
                'Fold 2': '{:.4f}',
                'Fold 3': '{:.4f}',
                'Fold 4': '{:.4f}',
                'Fold 5': '{:.4f}',
                'Std Dev': '{:.4f}'
            }).highlight_max(subset=['Best CV Score'], color='lightgreen'),
            use_container_width=True,
            hide_index=True
        )

        # Grafik per fold untuk BASE MODELS
        st.markdown("### Akurasi per Fold - Base Models")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        models_data = [
            ('Logistic Regression', [0.7601, 0.7089, 0.7466, 0.7655, 0.7595], 'skyblue'),
            ('KNN', [0.9461, 0.9380, 0.9326, 0.9434, 0.9622], 'lightgreen'),
            ('SVM', [0.9326, 0.9218, 0.9057, 0.9030, 0.9270], 'orange'),
            ('MLP', [0.8113, 0.8329, 0.8329, 0.8194, 0.8486], 'lightcoral')
        ]

        for i, (model_name, scores, color) in enumerate(models_data):
            folds = [f'Fold {j+1}' for j in range(5)]
            bars = axes[i].bar(folds, scores, color=color, alpha=0.8)
            axes[i].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Akurasi')
            axes[i].set_ylim(0.6, 1.0)
            axes[i].grid(True, alpha=0.3)

            # Tambah nilai di atas bar
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{score:.4f}', ha='center', va='bottom', fontsize=9)

            # Tambah garis rata-rata
            mean_score = np.mean(scores)
            axes[i].axhline(y=mean_score, color='red', linestyle='--', alpha=0.7, linewidth=2)
            axes[i].text(2, mean_score + 0.01, f'Mean: {mean_score:.4f}',
                        ha='center', va='bottom', color='red', fontweight='bold')

        plt.suptitle('Performa Akurasi Base Models per Fold CV', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

        # Tampilkan tabel META LEARNERS
        st.markdown("### Meta Learners Performance")
        meta_df = pd.DataFrame(meta_learners_results)

        st.dataframe(
            meta_df.style.format({
                'Best CV Score': '{:.4f}',
                'Fold 1': '{:.4f}',
                'Fold 2': '{:.4f}',
                'Fold 3': '{:.4f}',
                'Fold 4': '{:.4f}',
                'Fold 5': '{:.4f}',
                'Std Dev': '{:.4f}'
            }).highlight_max(subset=['Best CV Score'], color='lightgreen'),
            use_container_width=True,
            hide_index=True
        )

        # Grafik per fold untuk META LEARNERS
        st.markdown("### Akurasi per Fold - Meta Learners")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        meta_data = [
            ('KNN Meta', [0.9542, 0.9488, 0.9326, 0.9326, 0.9595], 'gold'),
            ('SVM Meta', [0.9515, 0.9434, 0.9353, 0.9380, 0.9568], 'lightblue'),
            ('MLP Meta', [0.9596, 0.9434, 0.9407, 0.9272, 0.9649], 'lightgreen'),
            ('LogReg Meta', [0.9542, 0.9380, 0.9380, 0.9326, 0.9568], 'plum')
        ]

        for i, (model_name, scores, color) in enumerate(meta_data):
            folds = [f'Fold {j+1}' for j in range(5)]
            bars = axes[i].bar(folds, scores, color=color, alpha=0.8)
            axes[i].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Akurasi')
            axes[i].set_ylim(0.92, 0.97)
            axes[i].grid(True, alpha=0.3)

            # Tambah nilai di atas bar
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{score:.4f}', ha='center', va='bottom', fontsize=9)

            # Tambah garis rata-rata
            mean_score = np.mean(scores)
            axes[i].axhline(y=mean_score, color='red', linestyle='--', alpha=0.7, linewidth=2)
            axes[i].text(2, mean_score + 0.002, f'Mean: {mean_score:.4f}',
                        ha='center', va='bottom', color='red', fontweight='bold')

        plt.suptitle('Performa Akurasi Meta Learners per Fold CV', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

          # Perbandingan semua model
        st.markdown("### Perbandingan Semua Model")
        fig, ax = plt.subplots(figsize=(14, 8))

        # Combine data
        all_models = list(base_df['Model']) + list(meta_df['Meta Learner'])
        all_scores = list(base_df['Best CV Score']) + list(meta_df['Best CV Score'])
        all_colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral', 'gold', 'lightblue', 'lime', 'plum']

        bars = ax.bar(all_models, all_scores, color=all_colors, alpha=0.8)
        ax.set_ylabel('Cross Validation Accuracy')
        ax.set_title('Perbandingan Akurasi: Base Models vs Meta Learners', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)  # Tambah ruang di atas
        ax.grid(True, alpha=0.3)

        # Tambah garis pemisah
        ax.axvline(x=3.5, color='red', linestyle='--', alpha=0.7, linewidth=2)

        # Pindahkan label ke posisi yang tidak overlap
        ax.text(1.5, 0.65, 'BASE MODELS', ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        ax.text(6, 0.65, 'META LEARNERS', ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

        # Tambah nilai di atas bar dengan posisi yang aman
        for bar, score in zip(bars, all_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                   f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)



        # ========== BAGIAN TESTING ==========

        st.markdown("## Evaluasi Model")

        dummy_results = [
            {"Model": "LogReg",          "Akurasi": 0.7200, "Presisi": 0.5616, "Recall": 0.9289, "Skor-F1": 0.7000},
            {"Model": "KNN",             "Akurasi": 0.9433, "Presisi": 0.9005, "Recall": 0.9431, "Skor-F1": 0.9213},
            {"Model": "SVM",             "Akurasi": 0.9050, "Presisi": 0.8407, "Recall": 0.9005, "Skor-F1": 0.8696},
            {"Model": "MLP",             "Akurasi": 0.7950, "Presisi": 0.6538, "Recall": 0.8863, "Skor-F1": 0.7525},
            {"Model": "Stacking_KNN",    "Akurasi": 0.9567, "Presisi": 0.9263, "Recall": 0.9526, "Skor-F1": 0.9393},
            {"Model": "Stacking_SVM",    "Akurasi": 0.9433, "Presisi": 0.9005, "Recall": 0.9431, "Skor-F1": 0.9213},
            {"Model": "Stacking_MLP",    "Akurasi": 0.9600, "Presisi": 0.9431, "Recall": 0.9431, "Skor-F1": 0.9431},
            {"Model": "Stacking_LogReg", "Akurasi": 0.9450, "Presisi": 0.9120, "Recall": 0.9336, "Skor-F1": 0.9227},
        ]

        df_perbandingan = pd.DataFrame(dummy_results)

        st.subheader("Evaluasi per Metode")
        st.dataframe(df_perbandingan.style.format({
            "Akurasi": "{:.4f}",
            "Presisi": "{:.4f}",
            "Recall": "{:.4f}",
            "Skor-F1": "{:.4f}",
        }), use_container_width=True)

        st.subheader("Perbandingan Metrik Evaluasi")
        for c in ["Akurasi", "Presisi", "Recall", "Skor-F1"]:
            df_perbandingan[c] = pd.to_numeric(df_perbandingan[c], errors="coerce")

        models = df_perbandingan["Model"].tolist()
        metrics = ["Akurasi", "Presisi", "Recall", "Skor-F1"]
        colors = ['skyblue']*4 + ['orange']*4  # Base = biru, Stacking = orange

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.ravel()

        for i, metrik in enumerate(metrics):
            vals = df_perbandingan[metrik].values
            bars = axes[i].bar(models, vals, color=colors, alpha=0.8)

            axes[i].set_title(f'Perbandingan {metrik}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(metrik)
            axes[i].set_ylim(0, 1.05)
            axes[i].grid(axis='y', alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)

            # Tambah nilai di atas bar
            for bar, v in zip(bars, vals):
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.4f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold'
                )

            # Garis pemisah Base vs Stacking
            axes[i].axvline(x=3.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

        plt.suptitle('Evaluasi Model pada Data Testing', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)


        # ========== CONFUSION MATRIX ==========
        st.markdown("### Confusion Matrix per Model")

        # Data confusion matrix (dari gambar yang Anda berikan)
        cm_data = {
            'LogReg': [[236, 153], [15, 196]],
            'KNN': [[367, 22], [12, 199]],
            'SVM': [[353, 36], [21, 190]],
            'MLP': [[290, 99], [24, 187]],
            'Stacking_KNN': [[373, 16], [10, 201]],
            'Stacking_SVM': [[367, 22], [12, 199]],
            'Stacking_MLP': [[377, 12], [12, 199]],
            'Stacking_LogReg': [[370, 19], [14, 197]]
        }

        # Buat 8 subplot (2 baris x 4 kolom)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()

        for i, (model_name, cm) in enumerate(cm_data.items()):
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=axes[i],
                xticklabels=['Non Diabetes', 'Diabetes'],
                yticklabels=['Non Diabetes', 'Diabetes'],
                cbar_kws={'shrink': 0.8}
            )

            # Tambahkan label Base/Stacking
            label = 'Base' if 'Stacking' not in model_name else 'Stacking'
            axes[i].set_title(f'{model_name} ({label})', fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Prediksi', fontsize=10)
            axes[i].set_ylabel('Aktual', fontsize=10)

        plt.suptitle('Confusion Matrix - Semua Model (Dummy)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    elif selected_tab == "Testing":
        st.markdown("## Testing")

        # Form input
        with st.form("form_prediksi"):
            st.subheader("Input Data Pasien")

            col1, col2 = st.columns(2)

            with col1:
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1,
                                            help="Jumlah kehamilan")
                glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120, step=1,
                                        help="Kadar glukosa darah (mg/dL)")
                blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, step=1,
                                                help="Tekanan darah diastolik (mmHg)")
                skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1,
                                                help="Ketebalan lipatan kulit trisep (mm)")

            with col2:
                insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80, step=1,
                                        help="Kadar insulin serum (mu U/ml)")
                bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1,
                                    help="Body Mass Index (kg/m²)")
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.001,
                                    help="Fungsi silsilah diabetes")
                age = st.number_input("Age", min_value=1, max_value=120, value=33, step=1,
                                    help="Umur (tahun)")

            submit_button = st.form_submit_button("Prediksi", use_container_width=True)

        if submit_button:
            # Data input
            input_data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age
            }

            st.markdown("---")
            st.subheader("Hasil Prediksi")

            # Tampilkan input
            with st.expander("Data Input Pasien", expanded=True):
                input_df = pd.DataFrame([input_data])
                st.dataframe(input_df, use_container_width=True)

            try:
                # Load models
                with st.spinner("Memuat model..."):
                    imputer = joblib.load('model_deployment/knn_imputer.pkl')
                    scaler = joblib.load('model_deployment/minmax_scaler.pkl')
                    lr_model = joblib.load('model_deployment/logistic_regression.pkl')
                    knn_model = joblib.load('model_deployment/knn.pkl')
                    svm_model = joblib.load('model_deployment/svm.pkl')
                    mlp_model = joblib.load('model_deployment/mlp.pkl')
                    meta_model = joblib.load('model_deployment/mlp_meta.pkl')
                    model_info = joblib.load('model_deployment/model_info.pkl')

                # Preprocessing
                with st.spinner("Preprocessing data..."):
                    df = pd.DataFrame([input_data])

                    # Imputasi (ubah 0 jadi NaN untuk kolom tertentu)
                    for col in model_info['kolom_imputasi']:
                        if df[col].iloc[0] == 0:
                            df[col] = np.nan

                    # Imputasi dengan KNN
                    df_imputed = pd.DataFrame(imputer.transform(df), columns=model_info['fitur'])

                    # Normalisasi
                    df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=model_info['fitur'])

                # Prediksi Base Models
                st.markdown("### Prediksi Base Models")
                st.caption("Setiap base model menghasilkan probabilitas untuk meta-learner")

                # Base models predict_proba - hasilnya shape (1, 2) untuk 1 sampel
                prediksi_lr = lr_model.predict_proba(df_scaled)   # Shape: (1, 2)
                prediksi_knn = knn_model.predict_proba(df_scaled) # Shape: (1, 2)
                prediksi_svm = svm_model.predict_proba(df_scaled) # Shape: (1, 2)
                prediksi_mlp = mlp_model.predict_proba(df_scaled) # Shape: (1, 2)

                # Tampilkan hasil base models (ambil [0] hanya untuk display)
                base_results = pd.DataFrame({
                    'Base Model': ['Logistic Regression', 'KNN', 'SVM', 'MLP'],
                    'Prob Non-Diabetes': [prediksi_lr[0][0], prediksi_knn[0][0], prediksi_svm[0][0], prediksi_mlp[0][0]],
                    'Prob Diabetes': [prediksi_lr[0][1], prediksi_knn[0][1], prediksi_svm[0][1], prediksi_mlp[0][1]]
                })

                st.dataframe(base_results.style.format({
                    'Prob Non-Diabetes': '{:.4f}',
                    'Prob Diabetes': '{:.4f}'
                }), use_container_width=True)



                # Meta Features - PERSIS SEPERTI DI COLAB
                st.markdown("### Meta Features")

                # Buat meta features dengan column_stack (seperti di colab)
                fitur_meta = np.column_stack([prediksi_lr, prediksi_knn, prediksi_svm, prediksi_mlp])
                # Shape fitur_meta: (1, 8) karena column_stack menggabungkan 4 array (1,2) menjadi (1,8)

                df_fitur_meta = pd.DataFrame(fitur_meta, columns=[
                    'lr_prob_0', 'lr_prob_1',      # LR probabilitas kelas 0 dan 1
                    'knn_prob_0', 'knn_prob_1',    # KNN probabilitas kelas 0 dan 1
                    'svm_prob_0', 'svm_prob_1',    # SVM probabilitas kelas 0 dan 1
                    'mlp_prob_0', 'mlp_prob_1'     # MLP probabilitas kelas 0 dan 1
                ])

                st.dataframe(df_fitur_meta.style.format("{:.4f}"), use_container_width=True)

                # Prediksi Final dengan Meta Learner (MLP)
                # ...existing code...
                # Prediksi Final dengan Meta Learner (MLP)
                st.markdown("### Prediksi Final (Meta Learner - MLP)")

                # Prediksi menggunakan meta learner
                final_pred = meta_model.predict(fitur_meta)[0]        # Ambil [0] karena hasil [[0]] atau [[1]]
                final_prob = meta_model.predict_proba(fitur_meta)[0]  # Ambil [0] karena hasil [[prob0, prob1]]

                # Tentukan hasil dan warna
                if final_pred == 0:
                    status = "Non-Diabetes"
                    bg_color = "#d4edda"  # Hijau muda
                    text_color = "#155724"  # Hijau tua
                else:
                    status = "Diabetes"
                    bg_color = "#f8d7da"  # Merah muda
                    text_color = "#721c24"  # Merah tua

                # Tampilkan hasil dengan styling custom
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border: 2px solid {text_color};">
                    <h2 style="color: {text_color}; text-align: center; margin: 0;">
                        Hasil Prediksi: {status}
                    </h2>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")  # Spacing

                # Tampilkan probabilitas
                col1, col2 = st.columns(2)




            except FileNotFoundError:
                st.error("❌ Model tidak ditemukan! Pastikan semua file model ada di folder `model_deployment/`")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")








