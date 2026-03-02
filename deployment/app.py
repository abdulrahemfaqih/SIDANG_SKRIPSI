import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import pages
from pages import data_page, preprocessing_page, modelling_page, testing_page

# Konfigurasi halaman
st.set_page_config(
    page_title="PENERAPAN METODE STACKING ENSEMBLE PADA KLASIFIKASI PENYAKIT DIABETES MELITUS",
    layout="wide"
)

# Sembunyikan default sidebar pages
st.markdown("""
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* Style untuk button menu */
    .stButton button {
        width: 100%;
        text-align: left;
        padding: 12px 20px;
        border-radius: 8px;
        border: none;
        margin-bottom: 8px;
        font-size: 16px;
        transition: all 0.3s;
    }

    .stButton button:hover {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("PENERAPAN METODE STACKING ENSEMBLE PADA KLASIFIKASI PENYAKIT DIABETES MELITUS")

# Inisialisasi session state untuk menyimpan halaman aktif
if 'active_page' not in st.session_state:
    st.session_state.active_page = "Data"

# Sidebar
st.sidebar.title("Menu")

# Menu buttons dengan status active
tabs = {
    "Data": "Data",
    "Preprocessing": "Preprocessing",
    "Modelling": "Modelling",
    "Testing": "Testing"
}

for label, page in tabs.items():
    # Tambahkan indikator active dengan warna button
    if st.session_state.active_page == page:
        button_type = "primary"
    else:
        button_type = "secondary"

    if st.sidebar.button(label, key=page, use_container_width=True, type=button_type):
        st.session_state.active_page = page
        st.rerun()

# Route ke halaman yang dipilih
if st.session_state.active_page == "Data":
    data_page.show()
elif st.session_state.active_page == "Preprocessing":
    preprocessing_page.show()
elif st.session_state.active_page == "Modelling":
    modelling_page.show()
elif st.session_state.active_page == "Testing":
    testing_page.show()