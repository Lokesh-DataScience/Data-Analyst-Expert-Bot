import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import requests
from langchain_core.messages import AIMessage, HumanMessage
import uuid
import base64
import time
from datetime import datetime, timedelta
from PIL import Image
import io
import hashlib
import pandas as pd
from utils.data_analyzer import DataAnalyzer
import plotly.express as px

# Set page config first, before any other Streamlit commands
st.set_page_config(page_title="Analyst Bot", layout="wide")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "image_upload_attempts" not in st.session_state:
    st.session_state.image_upload_attempts = []

if "last_uploaded_image_name" not in st.session_state:
    st.session_state.last_uploaded_image_name = None

@st.cache_data(ttl=120)  # cache for 2 minutes
def get_chat_history_for_session(session_id):
    try:
        res = requests.get(f"http://localhost:8000/recent-chats/{session_id}")
        res.raise_for_status()
        return res.json().get("chat_history", [])
    except Exception as e:
        st.warning(f"⚠️ Could not load recent chat history: {e}")
        return []

@st.cache_data(ttl=60)  # cache for 60 seconds
def get_recent_chat_titles():
    try:
        titles_res = requests.get("http://localhost:8000/recent-chat-titles")
        titles_res.raise_for_status()
        return titles_res.json().get("sessions", [])
    except Exception as e:
        st.sidebar.error(f"Failed to load recent chats: {e}")
        return []

def encode_image_file(file):
    """Encodes an image file to base64."""
    return base64.b64encode(file.read()).decode('utf-8')

def load_chat_history_from_backend(session_id):
    """Load and format chat history from backend"""
    full_history = get_chat_history_for_session(session_id)
    formatted_history = []
    
    for msg in full_history:
        if msg["type"] == "ai":
            formatted_history.append({
                "type": "ai",
                "content": msg["content"]
            })
        elif msg["type"] == "human":
            formatted_history.append({
                "type": "human",
                "content": msg["content"],
                "image": msg.get("file", {}).get("base64") if "file" in msg else msg.get("image"),
                "image_type": msg.get("file", {}).get("format") if "file" in msg else msg.get("image_type")
            })
    return formatted_history

# Load recent chat history from the backend on first load
if not st.session_state["chat_history"]:
    st.session_state["chat_history"] = load_chat_history_from_backend(st.session_state["session_id"])

# Clear old image upload timestamps (older than 6 hours)
cutoff_time = datetime.now() - timedelta(hours=6)
st.session_state.image_upload_attempts = [
    ts for ts in st.session_state.image_upload_attempts if ts > cutoff_time
]

# Check if image upload limit is reached
image_upload_limit_reached = len(st.session_state.image_upload_attempts) >= 3

st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Powered Data Analysis Platform</h1>
        <p>Upload your dataset and let AI handle the complete analysis pipeline</p>
    </div>
    """, unsafe_allow_html=True)
with st.sidebar:
    st.header("⚙️ Settings")
    show_raw_data = st.checkbox("Show raw data preview", value=False)
    max_rows_display = st.slider("Max rows to display", 5, 50, 10)
    analysis_depth = st.selectbox("Analysis Depth", ["Basic", "Detailed", "Advanced"], index=1)

# File upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.subheader("📂 Upload Your Dataset")
    st.markdown("**Supported formats:** CSV, Excel (.xlsx, .xls)")
    
    data_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        help="Upload your dataset for comprehensive analysis"
    )
    
    if data_file:
        file_details = {
            "Filename": data_file.name,
            "File size": f"{data_file.size / 1024:.2f} KB",
            "File type": data_file.type
        }
        
        st.markdown("**File Details:**")
        for key, value in file_details.items():
            st.write(f"• **{key}:** {value}")

st.markdown('</div>', unsafe_allow_html=True)


analyzer = DataAnalyzer()
if data_file:
    analyze_button = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
    
    if analyze_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load data
            status_text.text("📖 Loading dataset...")
            progress_bar.progress(10)
            
            if data_file.name.endswith(".csv"):
                # Try different encodings for CSV files
                try:
                    df = pd.read_csv(data_file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(data_file, encoding='latin-1')
                    except UnicodeDecodeError:
                        df = pd.read_csv(data_file, encoding='cp1252')
            else:
                df = pd.read_excel(data_file, engine='openpyxl')
            
            progress_bar.progress(30)
            
            # Display success message with dataset info
            st.markdown(f'''
            <div class="success-box">
                <h4>✅ Dataset loaded successfully!</h4>
                <p><strong>Shape:</strong> {df.shape[0]:,} rows × {df.shape[1]} columns</p>
                <p><strong>Size:</strong> {df.memory_usage(deep=True).sum() / 1024:.2f} KB</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("📈 Columns", df.shape[1])
            with col3:
                st.metric("🔢 Numeric Cols", df.select_dtypes(include=['number']).shape[1])
            with col4:
                st.metric("📝 Text Cols", df.select_dtypes(include=['object']).shape[1])
            
            # Show raw data if requested
            if show_raw_data:
                with st.expander("👁️ Raw Data Preview", expanded=False):
                    st.dataframe(df.head(max_rows_display), use_container_width=True)
            
            # Step 2: Data cleaning
            status_text.text("🧹 Cleaning dataset...")
            progress_bar.progress(50)
            
            cleaned_df, cleaning_log = analyzer.deep_clean_data(df)
            
            # Display cleaning results
            st.subheader("🧹 Data Cleaning Results")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Cleaning Steps Performed:**")
                for i, step in enumerate(cleaning_log, 1):
                    st.write(f"{i}. {step}")
            
            with col2:
                st.markdown("**Before vs After:**")
                comparison_data = {
                    "Metric": ["Rows", "Columns", "Missing Values", "Duplicates"],
                    "Before": [
                        df.shape[0],
                        df.shape[1],
                        df.isnull().sum().sum(),
                        df.duplicated().sum()
                    ],
                    "After": [
                        cleaned_df.shape[0],
                        cleaned_df.shape[1],
                        cleaned_df.isnull().sum().sum(),
                        cleaned_df.duplicated().sum()
                    ]
                }
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Cleaned data preview
            with st.expander("✨ Cleaned Data Preview", expanded=False):
                st.dataframe(cleaned_df.head(max_rows_display), use_container_width=True)
            
            # Download button for cleaned data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_buffer = io.StringIO()
            cleaned_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="💾 Download Cleaned Dataset",
                data=csv_data,
                file_name=f"cleaned_dataset_{timestamp}.csv",
                mime="text/csv",
                help="Download the cleaned version of your dataset"
            )
            
            progress_bar.progress(70)
            
            # Step 3: Statistical analysis
            status_text.text("📊 Performing statistical analysis...")
            
            st.subheader("📊 Statistical Summary")
            
            # Basic statistics
            numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                stats_summary = cleaned_df[numeric_cols].describe()
                st.markdown("#### 📈 Descriptive Statistics")
                st.dataframe(stats_summary, use_container_width=True)
                
                # Correlation matrix
                if len(numeric_cols) > 1:
                    st.markdown("#### 🔗 Correlation Analysis")
                    corr_matrix = cleaned_df[numeric_cols].corr()
                    
                    # Create correlation heatmap
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Correlation Matrix Heatmap",
                        color_continuous_scale="RdBu_r"
                    )
                    fig_corr.update_traces(texttemplate="%{z:.2f}", textfont_size=10)
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            # Data types and missing values analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Data Types")
                dtype_df = pd.DataFrame({
                    'Column': cleaned_df.columns,
                    'Data Type': cleaned_df.dtypes.astype(str),
                    'Non-Null Count': cleaned_df.count(),
                    'Null Count': cleaned_df.isnull().sum()
                })
                st.dataframe(dtype_df, use_container_width=True, hide_index=True)
            
            with col2:
                if cleaned_df.isnull().sum().sum() > 0:
                    st.markdown("#### ❌ Missing Values")
                    missing_data = cleaned_df.isnull().sum()
                    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                    
                    fig_missing = px.bar(
                        x=missing_data.index,
                        y=missing_data.values,
                        title="Missing Values by Column"
                    )
                    st.plotly_chart(fig_missing, use_container_width=True)
                else:
                    st.success("🎉 No missing values found!")
            
            progress_bar.progress(85)
            
            # Step 4: Visualizations
            status_text.text("📊 Creating visualizations...")
            
            st.subheader("📊 Data Visualizations")
            
            try:
                plots = analyzer.create_visualizations(cleaned_df)
                
                # Display plots in a grid layout
                plot_names = list(plots.keys())
                for i in range(0, len(plot_names), 2):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if i < len(plot_names):
                            st.plotly_chart(plots[plot_names[i]], use_container_width=True)
                    
                    with col2:
                        if i + 1 < len(plot_names):
                            st.plotly_chart(plots[plot_names[i + 1]], use_container_width=True)
                        
            except Exception as e:
                st.warning(f"⚠️ Could not generate all visualizations: {str(e)}")
                # Create basic visualizations as fallback
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    fig = px.histogram(cleaned_df, x=col, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            progress_bar.progress(95)
            
            # Step 5: AI Insights
            status_text.text("🤖 Generating AI insights...")
            
            st.subheader("🤖 AI-Generated Insights")
            
            with st.spinner("Analyzing patterns and generating insights..."):
                try:
                    insights = analyzer.generate_insights(cleaned_df, cleaning_log)
                    
                    # Format insights nicely
                    st.markdown(f'''
                    <div class="info-card">
                        {insights.replace(chr(10), '<br>')}
                    </div>
                    ''', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not generate AI insights: {str(e)}")
                    
                    # Provide basic insights as fallback
                    basic_insights = f"""
                    **Basic Dataset Insights:**
                    
                    • Your dataset contains **{cleaned_df.shape[0]:,} records** and **{cleaned_df.shape[1]} features**
                    • **{len(numeric_cols)} numeric columns** and **{len(cleaned_df.select_dtypes(include=['object']).columns)} text columns**
                    • Data cleaning removed **{df.shape[0] - cleaned_df.shape[0]} rows** and addressed **{len(cleaning_log)} issues**
                    • The dataset appears to be **{'well-structured' if cleaned_df.isnull().sum().sum() == 0 else 'moderately clean'}**
                    """
                    
                    st.markdown(f'''
                    <div class="info-card">
                        {basic_insights}
                    </div>
                    ''', unsafe_allow_html=True)
            
            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Success message
            st.success("🎉 Dataset analysis completed successfully!")
            
        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {str(e)}")
            st.info("💡 Please check your file format and try again. Ensure your data is properly formatted.")
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                st.write("**Error Details:**")
                st.code(str(e))
                
                if 'df' in locals():
                    st.write("**Dataset Info:**")
                    st.write(f"Shape: {df.shape}")
                    st.write(f"Columns: {list(df.columns)}")
                    st.write(f"Data types: {df.dtypes.to_dict()}")
# Sidebar for recent chats
with st.sidebar:
    st.header("🕓 Recent Chats")
    
    # New Chat button
    if st.button("➕ New Chat", type="primary", use_container_width=True):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["chat_history"] = []
        st.session_state["last_uploaded_image_name"] = None
        st.rerun()
    
    st.divider()
    
    session_titles = get_recent_chat_titles()
    
    for chat in session_titles:
        if st.button(chat["title"][:40], key=chat["session_id"]):
            st.session_state["session_id"] = chat["session_id"]
            st.session_state["chat_history"] = load_chat_history_from_backend(chat["session_id"])
            st.rerun()

# Main app title
st.title("💬 Data Analyst Expert")

# File upload sections
uploaded_image = None
if image_upload_limit_reached:
    st.warning("⏳ You've reached the image upload limit (3 per 6 hours). Try again later.")
else:
    uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
    
    # Track new image uploads
    if uploaded_image and uploaded_image.name != st.session_state.last_uploaded_image_name:
        st.session_state.image_upload_attempts.append(datetime.now())
        st.session_state.last_uploaded_image_name = uploaded_image.name

uploads_left = max(0, 3 - len(st.session_state.image_upload_attempts))
st.caption(f"🖼️ Uploads remaining in this 6-hour window: {uploads_left}")

# CSV and PDF upload
uploaded_csv = st.file_uploader("Upload a CSV file (optional)", type=["csv"])
uploaded_pdf = st.file_uploader("Upload a PDF file (optional)", type=["pdf"])

# Prepare file data
csv_b64 = None
csv_filename = None
if uploaded_csv:
    csv_b64 = base64.b64encode(uploaded_csv.read()).decode('utf-8')
    csv_filename = uploaded_csv.name

pdf_b64 = None
pdf_filename = None
if uploaded_pdf:
    pdf_b64 = base64.b64encode(uploaded_pdf.read()).decode('utf-8')
    pdf_filename = uploaded_pdf.name

# Chat input
user_input = st.chat_input("Ask me anything about data analysis...")

# Process user input
if user_input:
    # Encode image if uploaded
    image_b64 = encode_image_file(uploaded_image) if uploaded_image else None
    image_type = uploaded_image.type if uploaded_image else None

    # Add user message to chat history
    st.session_state.chat_history.append({
        "type": "human",
        "content": user_input,
        "image": image_b64,
        "image_type": image_type
    })

    # Prepare chat history for API (serialize properly)
    history_serialized = []
    for msg in st.session_state.chat_history:
        if isinstance(msg, dict):
            history_serialized.append({
                "type": msg["type"],
                "content": msg["content"]
            })
        elif isinstance(msg, AIMessage):
            history_serialized.append({
                "type": "ai",
                "content": msg.content
            })
        elif isinstance(msg, HumanMessage):
            history_serialized.append({
                "type": "human",
                "content": msg.content
            })

    try:
        with st.spinner("Thinking..."):
            # Prepare payload
            payload = {
                "question": user_input,
                "session_id": st.session_state["session_id"],
                "chat_history": history_serialized
            }
            
            # Determine API endpoint based on uploaded files
            if csv_b64 and csv_filename:
                payload["csv_base64"] = csv_b64
                payload["csv_filename"] = csv_filename
                api_url = "http://localhost:8000/csv-upload"
            elif image_b64:
                payload["image_base64"] = image_b64
                payload["image_type"] = image_type
                api_url = "http://localhost:8000/image-upload"
            elif pdf_b64 and pdf_filename:
                payload["pdf_base64"] = pdf_b64
                payload["pdf_filename"] = pdf_filename
                api_url = "http://localhost:8000/pdf-upload"
            else:
                api_url = "http://localhost:8000/chat"

            # Send request to API
            res = requests.post(api_url, json=payload)
            res.raise_for_status()
            
            answer = res.json().get("response", "⚠️ No answer returned.")
            
            # Add AI response to chat history
            st.session_state.chat_history.append({
                "type": "ai",
                "content": answer
            })

    except requests.exceptions.RequestException as e:
        error_msg = f"❌ API Error: {e}"
        st.session_state.chat_history.append({
            "type": "ai",
            "content": error_msg
        })
    except Exception as e:
        error_msg = f"❌ Unexpected Error: {e}"
        st.session_state.chat_history.append({
            "type": "ai",
            "content": error_msg
        })

# Display chat history
shown_images = set()

for msg in st.session_state.chat_history:
    if isinstance(msg, AIMessage):
        st.chat_message("ai").markdown(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, dict):
        if msg.get("type") == "ai":
            st.chat_message("ai").markdown(msg["content"])
        elif msg.get("type") == "human":
            with st.chat_message("user"):
                st.markdown(msg["content"])
                
                # Display image if present
                if msg.get("image"):
                    try:
                        image_data = base64.b64decode(msg["image"])
                        image_hash = hashlib.md5(image_data).hexdigest()
                        
                        # Only show each unique image once
                        if image_hash not in shown_images:
                            shown_images.add(image_hash)
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, caption="Uploaded Image", use_container_width=True)
                    except Exception as e:
                        st.error(f"🖼️ Could not display image: {e}")