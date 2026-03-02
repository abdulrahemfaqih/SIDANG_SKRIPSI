import streamlit as st
import joblib

@st.cache_resource
def load_models():
    try:
        imputer = joblib.load('model_deployment/knn_imputer.pkl')
        scaler = joblib.load('model_deployment/minmax_scaler.pkl')

        lr_model = joblib.load('model_deployment/logistic_regression.pkl')
        knn_model = joblib.load('model_deployment/knn.pkl')
        svm_model = joblib.load('model_deployment/svm.pkl')
        mlp_model = joblib.load('model_deployment/mlp.pkl')

        meta_model = joblib.load('model_deployment/mlp_meta.pkl')
        info = joblib.load('model_deployment/model_info.pkl')

        return imputer, scaler, lr_model, knn_model, svm_model, mlp_model, meta_model, info
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None