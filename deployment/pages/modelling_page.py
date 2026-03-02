import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from utils.data_loader import load_preprocessed_data

def show():
    st.header("Training Model")

    st.subheader("Arsitektur Stacking Ensemble")
    st.markdown("""
    **Base Models (Level 1):**
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Multi-Layer Perceptron (MLP)

    **Meta Learner (Level 2):**
    Setiap base model dicoba sebagai meta learner:
    - Skenario 1: KNN sebagai meta learner
    - Skenario 2: SVM sebagai meta learner
    - Skenario 3: MLP sebagai meta learner
    - Skenario 4: Logistic Regression sebagai meta learner
    """)

    # Cross validation
    st.subheader("Cross Validation Setup")
    st.markdown("""
    **K-Fold Cross Validation dengan 5 fold** pada data yang sudah dipreprocessing lengkap:
    - Dataset dibagi menjadi 5 fold secara acak
    - Model dilatih pada 4 fold dan ditest pada 1 fold
    - Proses diulang 5 kali dengan fold berbeda
    """)

    X_final, y_final, X_test_final, y_test_final = load_preprocessed_data()

    if X_final is None:
        return

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_info = []
    fold_num = 1

    for train_idx, test_idx in kf.split(X_final):
        train_size = len(train_idx)
        test_size = len(test_idx)

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

    st.subheader("Detail Cross Validation Splits")
    cv_df = pd.DataFrame(cv_info)

    # Tabel CV matrix
    total_samples = len(X_final)
    samples_per_fold = total_samples // 5
    remaining = total_samples % 5

    fold_sizes = []
    for i in range(5):
        if i < remaining:
            fold_sizes.append(samples_per_fold + 1)
        else:
            fold_sizes.append(samples_per_fold)

    cv_matrix_data = []
    for iterasi in range(1, 6):
        row = {'Iterasi': f'Iterasi {iterasi}'}
        for fold in range(1, 6):
            if fold == iterasi:
                row[f'Fold {fold}'] = f"{fold_sizes[fold-1]} (Val)"
            else:
                row[f'Fold {fold}'] = f"{fold_sizes[fold-1]} (Train)"
        cv_matrix_data.append(row)

    cv_matrix_df = pd.DataFrame(cv_matrix_data)

    def highlight_validation(val):
        if '(Val)' in str(val):
            return 'background-color: #ffeb3b; font-weight: bold'
        elif '(Train)' in str(val):
            return 'background-color: #e3f2fd'
        return ''

    st.dataframe(
        cv_matrix_df.style.applymap(highlight_validation),
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🟨 **Validation Set** - Data untuk evaluasi model")
    with col2:
        st.markdown("🟦 **Training Set** - Data untuk melatih model")

    # Visualisasi CV
    _show_cv_visualization(cv_df)

    # Hyperparameter
    _show_hyperparameters()

    # Training results
    _show_training_results()

    # Evaluation
    _show_evaluation()

def _show_cv_visualization(cv_df):
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
        ax.set_title('Pembagian Data Training-Validation per Iterasi CV')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Iterasi {i+1}' for i in range(5)])
        ax.legend()
        ax.grid(True, alpha=0.3)

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

        train_non_diabetes = cv_df['Train_Non_Diabetes'].values
        train_diabetes = cv_df['Train_Diabetes'].values

        x = np.arange(len(cv_df))

        bars1 = ax.bar(x, train_non_diabetes, label='Non Diabetes',
                      color='lightblue', alpha=0.8)
        bars2 = ax.bar(x, train_diabetes, bottom=train_non_diabetes,
                      label='Diabetes', color='lightcoral', alpha=0.8)

        ax.set_xlabel('Iterasi Cross Validation')
        ax.set_ylabel('Jumlah Sampel Training')
        ax.set_title('Distribusi Kelas dalam Data Training per Iterasi CV')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Iterasi {i+1}' for i in range(5)])
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

def _show_hyperparameters():
    st.subheader("Hyperparameter Tuning")
    st.markdown("""
    Setiap model menggunakan **GridSearchCV** dengan 5-fold cross validation.
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

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Logistic Regression")
        for param, values in param_info['Logistic Regression'].items():
            st.markdown(f"**{param}**: `{values}`")

        st.markdown("### K-Nearest Neighbors (KNN)")
        for param, values in param_info['KNN'].items():
            st.markdown(f"**{param}**: `{values}`")

    with col2:
        st.markdown("### Support Vector Machine (SVM)")
        for param, values in param_info['SVM'].items():
            st.markdown(f"**{param}**: `{values}`")

        st.markdown("### Multi-Layer Perceptron (MLP)")
        for param, values in param_info['MLP'].items():
            st.markdown(f"**{param}**: `{values}`")

def _show_training_results():
    st.subheader("Hasil Training Model")

    base_models_results = {
        'Model': ['Logistic Regression', 'KNN', 'SVM', 'MLP'],
        'Best CV Score': [0.7481, 0.9445, 0.9180, 0.8290],
        'Fold 1': [0.7601, 0.9461, 0.9326, 0.8113],
        'Fold 2': [0.7089, 0.9380, 0.9218, 0.8329],
        'Fold 3': [0.7466, 0.9326, 0.9057, 0.8329],
        'Fold 4': [0.7655, 0.9434, 0.9030, 0.8194],
        'Fold 5': [0.7595, 0.9622, 0.9270, 0.8486],
        'Std Dev': [0.0206, 0.0100, 0.0117, 0.0128]
    }

    meta_learners_results = {
        'Meta Learner': ['KNN Meta', 'SVM Meta', 'MLP Meta', 'LogReg Meta'],
        'Best CV Score': [0.9455, 0.9450, 0.9472, 0.9439],
        'Fold 1': [0.9542, 0.9515, 0.9596, 0.9542],
        'Fold 2': [0.9488, 0.9434, 0.9434, 0.9380],
        'Fold 3': [0.9326, 0.9353, 0.9407, 0.9380],
        'Fold 4': [0.9326, 0.9380, 0.9272, 0.9326],
        'Fold 5': [0.9595, 0.9568, 0.9649, 0.9568],
        'Std Dev': [0.0111, 0.0081, 0.0136, 0.0097]
    }

    st.markdown("### Base Models Performance")
    base_df = pd.DataFrame(base_models_results)
    st.dataframe(base_df.style.format({col: '{:.4f}' for col in base_df.columns if col != 'Model'}).highlight_max(subset=['Best CV Score'], color='lightgreen'), use_container_width=True, hide_index=True)

    # Grafik base models
    _show_fold_charts_base(base_df)

    st.markdown("### Meta Learners Performance")
    meta_df = pd.DataFrame(meta_learners_results)
    st.dataframe(meta_df.style.format({col: '{:.4f}' for col in meta_df.columns if col != 'Meta Learner'}).highlight_max(subset=['Best CV Score'], color='lightgreen'), use_container_width=True, hide_index=True)

    # Grafik meta learners
    _show_fold_charts_meta(meta_df)

    # Perbandingan
    _show_comparison_chart(base_df, meta_df)

def _show_fold_charts_base(base_df):
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

        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{score:.4f}', ha='center', va='bottom', fontsize=9)

        mean_score = np.mean(scores)
        axes[i].axhline(y=mean_score, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[i].text(2, mean_score + 0.01, f'Mean: {mean_score:.4f}',
                    ha='center', va='bottom', color='red', fontweight='bold')

    plt.suptitle('Performa Akurasi Base Models per Fold CV', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

def _show_fold_charts_meta(meta_df):
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

        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{score:.4f}', ha='center', va='bottom', fontsize=9)

        mean_score = np.mean(scores)
        axes[i].axhline(y=mean_score, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[i].text(2, mean_score + 0.002, f'Mean: {mean_score:.4f}',
                    ha='center', va='bottom', color='red', fontweight='bold')

    plt.suptitle('Performa Akurasi Meta Learners per Fold CV', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

def _show_comparison_chart(base_df, meta_df):
    st.markdown("### Perbandingan Semua Model")
    fig, ax = plt.subplots(figsize=(14, 8))

    all_models = list(base_df['Model']) + list(meta_df['Meta Learner'])
    all_scores = list(base_df['Best CV Score']) + list(meta_df['Best CV Score'])
    all_colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral', 'gold', 'lightblue', 'lime', 'plum']

    bars = ax.bar(all_models, all_scores, color=all_colors, alpha=0.8)
    ax.set_ylabel('Cross Validation Accuracy')
    ax.set_title('Perbandingan Akurasi: Base Models vs Meta Learners', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax.axvline(x=3.5, color='red', linestyle='--', alpha=0.7, linewidth=2)

    ax.text(1.5, 0.65, 'BASE MODELS', ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    ax.text(6, 0.65, 'META LEARNERS', ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

    for bar, score in zip(bars, all_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
               f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

def _show_evaluation():
    st.markdown("## Evaluasi Model")

    dummy_results = [
        {"Model": "LogReg", "Akurasi": 0.7200, "Presisi": 0.5616, "Recall": 0.9289, "Skor-F1": 0.7000},
        {"Model": "KNN", "Akurasi": 0.9433, "Presisi": 0.9005, "Recall": 0.9431, "Skor-F1": 0.9213},
        {"Model": "SVM", "Akurasi": 0.9050, "Presisi": 0.8407, "Recall": 0.9005, "Skor-F1": 0.8696},
        {"Model": "MLP", "Akurasi": 0.7950, "Presisi": 0.6538, "Recall": 0.8863, "Skor-F1": 0.7525},
        {"Model": "Stacking_KNN", "Akurasi": 0.9567, "Presisi": 0.9263, "Recall": 0.9526, "Skor-F1": 0.9393},
        {"Model": "Stacking_SVM", "Akurasi": 0.9433, "Presisi": 0.9005, "Recall": 0.9431, "Skor-F1": 0.9213},
        {"Model": "Stacking_MLP", "Akurasi": 0.9600, "Presisi": 0.9431, "Recall": 0.9431, "Skor-F1": 0.9431},
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
    colors = ['skyblue']*4 + ['orange']*4

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

        for bar, v in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        axes[i].axvline(x=3.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

    plt.suptitle('Evaluasi Model pada Data Testing', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    # Confusion Matrix
    _show_confusion_matrices()

def _show_confusion_matrices():
    st.markdown("### Confusion Matrix per Model")

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

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for i, (model_name, cm) in enumerate(cm_data.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Non Diabetes', 'Diabetes'],
                   yticklabels=['Non Diabetes', 'Diabetes'],
                   cbar_kws={'shrink': 0.8})

        label = 'Base' if 'Stacking' not in model_name else 'Stacking'
        axes[i].set_title(f'{model_name} ({label})', fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Prediksi', fontsize=10)
        axes[i].set_ylabel('Aktual', fontsize=10)

    plt.suptitle('Confusion Matrix - Semua Model', fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)