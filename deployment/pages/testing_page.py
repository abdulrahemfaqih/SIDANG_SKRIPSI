import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils.model_loader import load_models

def show():
    st.markdown("## Testing")

    with st.form("form_prediksi"):
        st.subheader("Input Data Pasien")

        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1)
            glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120, step=1)
            blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, step=1)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)

        with col2:
            insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80, step=1)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.001)
            age = st.number_input("Age", min_value=1, max_value=120, value=33, step=1)

        submit_button = st.form_submit_button("Prediksi", use_container_width=True)

    if submit_button:
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

        with st.expander("Data Input Pasien", expanded=True):
            input_df = pd.DataFrame([input_data])
            st.dataframe(input_df, use_container_width=True)

        try:
            with st.spinner("Memuat model..."):
                models = load_models()
                if models is None:
                    return

                imputer, scaler, lr_model, knn_model, svm_model, mlp_model, meta_model, model_info = models

            with st.spinner("Preprocessing data..."):
                df = pd.DataFrame([input_data])

                for col in model_info['kolom_imputasi']:
                    if df[col].iloc[0] == 0:
                        df[col] = np.nan

                df_imputed = pd.DataFrame(imputer.transform(df), columns=model_info['fitur'])
                df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=model_info['fitur'])

            st.markdown("### Prediksi Base Models")
            prediksi_lr = lr_model.predict_proba(df_scaled)
            prediksi_knn = knn_model.predict_proba(df_scaled)
            prediksi_svm = svm_model.predict_proba(df_scaled)
            prediksi_mlp = mlp_model.predict_proba(df_scaled)

            base_results = pd.DataFrame({
                'Base Model': ['Logistic Regression', 'KNN', 'SVM', 'MLP'],
                'Prob Non-Diabetes': [prediksi_lr[0][0], prediksi_knn[0][0], prediksi_svm[0][0], prediksi_mlp[0][0]],
                'Prob Diabetes': [prediksi_lr[0][1], prediksi_knn[0][1], prediksi_svm[0][1], prediksi_mlp[0][1]]
            })

            st.dataframe(base_results.style.format({
                'Prob Non-Diabetes': '{:.4f}',
                'Prob Diabetes': '{:.4f}'
            }), use_container_width=True)

            st.markdown("### Meta Features")
            fitur_meta = np.column_stack([prediksi_lr, prediksi_knn, prediksi_svm, prediksi_mlp])

            df_fitur_meta = pd.DataFrame(fitur_meta, columns=[
                'lr_prob_0', 'lr_prob_1',
                'knn_prob_0', 'knn_prob_1',
                'svm_prob_0', 'svm_prob_1',
                'mlp_prob_0', 'mlp_prob_1'
            ])

            st.dataframe(df_fitur_meta.style.format("{:.4f}"), use_container_width=True)

            st.markdown("### Prediksi Final (Meta Learner - MLP)")
            final_pred = meta_model.predict(fitur_meta)[0]
            final_prob = meta_model.predict_proba(fitur_meta)[0]

            if final_pred == 0:
                status = "Non-Diabetes"
                bg_color = "#d4edda"
                text_color = "#155724"
            else:
                status = "Diabetes"
                bg_color = "#f8d7da"
                text_color = "#721c24"

            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border: 2px solid {text_color};">
                <h2 style="color: {text_color}; text-align: center; margin: 0;">
                    Hasil Prediksi: {status}
                </h2>
            </div>
            """, unsafe_allow_html=True)

        except FileNotFoundError:
            st.error("❌ Model tidak ditemukan!")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {str(e)}")