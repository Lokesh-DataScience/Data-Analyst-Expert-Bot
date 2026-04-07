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
import base64

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
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Analysis", "📊 Data Upload & Analysis", "🛠️ SQL Query Generator", "🔧 Data Augmentation"])

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
    with tab3:
        st.subheader("🛠️ SQL Query Generator")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Database Schema (optional)**")
            schema_input = st.text_area(
                "Paste your table schema or DDL here",
                placeholder="CREATE TABLE orders (\n  id INT PRIMARY KEY,\n  customer_id INT,\n  total DECIMAL(10,2),\n  created_at TIMESTAMP\n);",
                height=200,
                key="sql_schema"
            )

        with col2:
            st.markdown("**Database Type**")
            db_type = st.selectbox(
                "Select your database",
                options=["PostgreSQL", "MySQL", "SQLite", "Microsoft SQL Server", "Oracle", "BigQuery", "Snowflake"],
                key="sql_db_type"
            )
            st.markdown("**Query Type**")
            query_type = st.selectbox(
                "What kind of query do you need?",
                options=["SELECT / Fetch Data", "INSERT / Add Data", "UPDATE / Modify Data", "DELETE / Remove Data", "JOIN / Combine Tables", "Aggregation / GROUP BY", "Subquery / CTE", "Other"],
                key="sql_query_type"
            )

        st.markdown("**Describe what you want to query**")
        sql_user_input = st.text_area(
            "Describe your query in plain English",
            placeholder="e.g. Find all customers who placed more than 3 orders in the last 30 days, sorted by total spend descending.",
            height=100,
            key="sql_description"
        )

        uploaded_schema_csv = st.file_uploader(
            "Or upload a CSV/Excel file to auto-detect schema",
            type=AppConfig.ALLOWED_CSV_TYPES,
            key="sql_schema_csv"
        )

        if uploaded_schema_csv:
            try:
                df_preview = pd.read_csv(uploaded_schema_csv) if uploaded_schema_csv.name.endswith(".csv") else pd.read_excel(uploaded_schema_csv)
                st.markdown("**Preview (first 5 rows):**")
                st.dataframe(df_preview.head(5), use_container_width=True)
                # Auto-generate schema hint from dataframe
                auto_schema = ", ".join([f"{col} ({str(dtype)})" for col, dtype in zip(df_preview.columns, df_preview.dtypes)])
                if not schema_input:
                    schema_input = f"-- Auto-detected columns:\n-- {auto_schema}"
            except Exception as e:
                st.warning(f"Could not preview file: {e}")

        generate_button = st.button("⚡ Generate SQL Query", type="primary", use_container_width=True)

        if generate_button:
            if not sql_user_input.strip():
                st.warning("⚠️ Please describe what you want to query.")
            else:
                try:
                    payload = {
                        "description": sql_user_input,
                        "schema": schema_input,
                        "db_type": db_type,
                        "query_type": query_type,
                        "session_id": session_manager.get_session_id()
                    }
                    with st.spinner("Generating SQL query..."):
                        response = api_client.generate_sql_query(payload)

                    if response:
                        st.success("✅ SQL Query Generated!")
                        st.markdown("**Generated Query:**")
                        st.code(response.get("sql_query", ""), language="sql")

                        if response.get("explanation"):
                            with st.expander("📖 Explanation"):
                                st.markdown(response["explanation"])

                        if response.get("suggestions"):
                            with st.expander("💡 Optimization Suggestions"):
                                st.markdown(response["suggestions"])

                        col_copy, col_download = st.columns([1, 1])
                        with col_download:
                            st.download_button(
                                label="⬇️ Download .sql file",
                                data=response.get("sql_query", ""),
                                file_name="generated_query.sql",
                                mime="text/plain",
                                use_container_width=True
                            )

                except Exception as e:
                    st.error(f"❌ An error occurred while generating the query: {e}")
        with tab4:
            st.subheader("🔧 Auto Data Augmentation")
            st.caption("Automatically detect and fix data quality issues before analysis.")

            aug_file = st.file_uploader(
                "Upload CSV file",
                type=AppConfig.ALLOWED_CSV_TYPES,
                key="aug_csv"
            )

            if aug_file:
                # Clear state when a new file is uploaded
                if st.session_state.get("aug_last_file") != aug_file.name:
                    st.session_state.pop("aug_diagnosis", None)
                    st.session_state.pop("aug_result", None)
                    st.session_state.aug_last_file = aug_file.name

                file_details = get_file_details(aug_file)
                st.table(pd.DataFrame(file_details.items(), columns=["Property", "Value"]))

                # ── STAGE 1: Diagnose ──────────────────────────────────────────
                if st.button("🔍 Diagnose Data", type="primary", use_container_width=True):
                    b64, filename = encode_csv_file(aug_file)
                    with st.spinner("Scanning your data for issues..."):
                        diagnosis = api_client.diagnose_data({
                            "csv_base64": b64,
                            "csv_filename": filename,
                            "session_id": session_manager.get_session_id()
                        })
                    st.session_state.aug_diagnosis = diagnosis
                    st.session_state.aug_b64 = b64
                    st.session_state.aug_filename = filename

                # ── Show diagnosis results ─────────────────────────────────────
                if "aug_diagnosis" in st.session_state:
                    diagnosis = st.session_state.aug_diagnosis

                    if diagnosis.get("has_issues"):
                        st.warning(f"⚠️ Found {len(diagnosis['issues'])} issue(s) in your dataset.")
                    else:
                        st.success("✅ No major issues detected.")

                    # Issues summary
                    for issue in diagnosis.get("issues", []):
                        st.markdown(f"- {issue}")

                    # Recommendations table
                    recs = diagnosis.get("recommendations", [])
                    if recs:
                        st.markdown("**📋 Augmentation Plan:**")
                        rec_df = pd.DataFrame([{
                            "Type": r["type"].replace("_", " ").title(),
                            "Description": r["description"],
                            "Severity": r.get("severity", "—").upper()
                        } for r in recs])
                        st.dataframe(rec_df, use_container_width=True, hide_index=True)

                    # ── STAGE 2: Options ───────────────────────────────────────
                    st.markdown("---")
                    st.markdown("**⚙️ Select Augmentation Options:**")

                    col1, col2 = st.columns(2)
                    with col1:
                        apply_imputation = st.checkbox("🩹 Impute Missing Values", value=True)
                        apply_outliers = st.checkbox("📐 Treat Outliers (Winsorize)", value=True)
                        apply_dedup = st.checkbox("🗑️ Remove Duplicates", value=True)
                    with col2:
                        apply_transform = st.checkbox("📈 Fix Skewed Distributions (Log)", value=False)
                        apply_synthetic = st.checkbox(
                            "🧬 Generate Synthetic Rows",
                            value=False,
                            help="Adds Gaussian-noise rows to expand small datasets. Use with caution."
                        )

                    if st.button("⚡ Apply Augmentation", type="primary", use_container_width=True):
                        with st.spinner("Augmenting your data..."):
                            result = api_client.augment_data({
                                "csv_base64": st.session_state.aug_b64,
                                "csv_filename": st.session_state.aug_filename,
                                "session_id": session_manager.get_session_id(),
                                "apply_imputation": apply_imputation,
                                "apply_outlier_treatment": apply_outliers,
                                "apply_deduplication": apply_dedup,
                                "apply_transformations": apply_transform,
                                "apply_synthetic_rows": apply_synthetic
                            })
                        st.session_state.aug_result = result

                # ── STAGE 3: Results ───────────────────────────────────────────
                if "aug_result" in st.session_state:
                    result = st.session_state.aug_result

                    if not result.get("success"):
                        st.error(f"❌ {result.get('message', 'Augmentation failed.')}")
                    else:
                        orig_rows, orig_cols = result["original_shape"]
                        aug_rows, aug_cols = result["augmented_shape"]

                        st.success("✅ Augmentation Complete!")

                        # Before / after metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Rows Before", orig_rows, delta=None)
                        m2.metric("Rows After", aug_rows, delta=f"+{aug_rows - orig_rows}" if aug_rows > orig_rows else str(aug_rows - orig_rows))
                        m3.metric("Columns", aug_cols)

                        # Change log
                        change_log = result.get("change_log", [])
                        if change_log:
                            with st.expander("📝 Change Log", expanded=True):
                                for entry in change_log:
                                    st.markdown(f"**{entry['step']}** — {entry['detail']}")

                        # Side-by-side preview
                        with st.expander("🔍 Data Preview (Before vs After)"):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**Original**")
                                st.dataframe(pd.DataFrame(result["sample_original"]), use_container_width=True)
                            with c2:
                                st.markdown("**Augmented**")
                                st.dataframe(pd.DataFrame(result["sample_augmented"]), use_container_width=True)

                        # Download augmented CSV
                        augmented_bytes = base64.b64decode(result["augmented_csv_base64"])
                        st.download_button(
                            label="⬇️ Download Augmented CSV",
                            data=augmented_bytes,
                            file_name=result["augmented_filename"],
                            mime="text/csv",
                            use_container_width=True
                        )

                        # Feed into analysis
                        st.markdown("---")
                        if st.button("📊 Run Analysis on Augmented Data", use_container_width=True):
                            with st.spinner("Analyzing augmented data..."):
                                analysis = api_client.analyze_data({
                                    "csv_base64": result["augmented_csv_base64"],
                                    "csv_filename": result["augmented_filename"],
                                    "session_id": session_manager.get_session_id()
                                })
                            st.session_state.aug_analysis = analysis

                        if "aug_analysis" in st.session_state:
                            display_analysis_results(st.session_state.aug_analysis)
    st.markdown(
        """
        <hr style="margin-top:2em;margin-bottom:0.5em;">
        <div style="text-align:center; color: #888; font-size: 0.95em;">
            &copy; 2025 <a href="https://lokesh-datascience.github.io/portfolio/" target="_blank" style="color:#888;text-decoration:underline;">Developed By Lokesh Kumar</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()