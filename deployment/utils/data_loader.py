import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    try:
        data = pd.read_csv('data/diabetes_frankurt_germany.csv')
        return data
    except:
        st.error("File diabetes_frankurt_germany.csv tidak ditemukan!")
        return None

@st.cache_data
def load_preprocessed_data():
    try:
        train_data = pd.read_csv('data/data_training_smote.csv')
        X_final = train_data.drop('Outcome', axis=1)
        y_final = train_data['Outcome']

        test_data = pd.read_csv('data/data_testing_normalized.csv')
        X_test_final = test_data.drop('Outcome', axis=1)
        y_test_final = test_data['Outcome']

        return X_final, y_final, X_test_final, y_test_final
    except FileNotFoundError:
        st.error("File CSV preprocessing tidak ditemukan!")
        return None, None, None, None