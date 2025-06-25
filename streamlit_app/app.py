import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from config.settings import AppConfig
from components.sidebar import render_sidebar
from components.file_upload import encode_image_file, encode_csv_file, encode_pdf_file, get_file_details
from components.chat_interface import render_chat_interface
from components.data_analysis import display_analysis_results
from utils.session_manager import SessionManager
from utils.api_client import APIClient
import pandas as pd
# Page configuration
st.set_page_config(
    page_title=AppConfig.PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point"""
    # Initialize session manager
    session_manager = SessionManager()
    session_manager.initialize_session()

    # API client
    api_client = APIClient()

    # Render main header
    st.markdown(f"""
        <div class="main-header">
            <h1>{AppConfig.APP_TITLE}</h1>
            <p>{AppConfig.APP_DESCRIPTION}</p>
        </div>
        """, unsafe_allow_html=True)

    # Render sidebar
    render_sidebar(
        api_client.get_recent_chat_titles,
        api_client.get_chat_history
    )

    # Main content area
    tab1, tab2 = st.tabs(["💬 Chat Analysis", "📊 Data Upload & Analysis"])

    with tab1:
        st.subheader("💬 Chat or Upload a File")

        uploaded_image = uploaded_csv = uploaded_pdf = None

        uploaded_image = st.file_uploader("Upload an image", type=AppConfig.ALLOWED_IMAGE_TYPES, key="chat_image")
        uploaded_csv = st.file_uploader("Upload a CSV/Excel file", type=AppConfig.ALLOWED_CSV_TYPES, key="chat_csv")
        uploaded_pdf = st.file_uploader("Upload a PDF", type=AppConfig.ALLOWED_PDF_TYPES, key="chat_pdf")

        user_input = st.chat_input("Ask a question about your data or upload a file...")

        render_chat_interface(
            user_input=user_input,
            uploaded_image=uploaded_image,
            uploaded_csv=uploaded_csv,
            uploaded_pdf=uploaded_pdf,
            encode_image_file=encode_image_file,
            load_chat_history_from_backend=api_client.get_chat_history,
            session_id=session_manager.get_session_id(),
            chat_history=session_manager.get_chat_history()
        )

    with tab2:
        st.subheader("📊 Upload a Dataset for Analysis")
        data_file = st.file_uploader("Upload CSV or Excel file", type=AppConfig.ALLOWED_CSV_TYPES, key="analysis_csv")
        if data_file:
            file_details = get_file_details(data_file)
            st.markdown("**File Details:**")
            st.table(pd.DataFrame(file_details.items(), columns=["Property", "Value"]))
            analyze_button = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
            if analyze_button:
                try:
                    b64, filename = encode_csv_file(data_file)
                    payload = {
                        "csv_base64": b64,
                        "csv_filename": filename,
                        "session_id": session_manager.get_session_id()
                    }
                    with st.spinner("Analyzing your data..."):
                        response = api_client.analyze_data(payload)
                    display_analysis_results(response)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()