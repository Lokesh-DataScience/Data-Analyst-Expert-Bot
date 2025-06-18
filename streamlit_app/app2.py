import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from utils.data_analyzer import DataAnalyzer
# Configure page
st.set_page_config(
    page_title="Data Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main header
st.markdown('<h1 class="main-header">📊 Data Analyzer Pro</h1>', unsafe_allow_html=True)

# Sidebar for additional options
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

# Initialize analyzer (assuming DataAnalyzer class exists)
try:
    analyzer = DataAnalyzer()
except NameError:
    # Fallback if DataAnalyzer class is not defined
    st.error("⚠️ DataAnalyzer class not found. Please ensure it's properly imported.")
    st.stop()

# Main analysis section
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
            st.balloons()
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

else:
    # Welcome message when no file is uploaded
    st.markdown("""
    <div class="info-card">
        <h3>👋 Welcome to Data Analyzer Pro!</h3>
        <p>Upload your dataset to get started with comprehensive data analysis including:</p>
        <ul>
            <li>🧹 <strong>Automated data cleaning</strong> - Remove duplicates, handle missing values</li>
            <li>📊 <strong>Statistical analysis</strong> - Descriptive statistics and correlations</li>
            <li>📈 <strong>Interactive visualizations</strong> - Charts and graphs to explore your data</li>
            <li>🤖 <strong>AI-powered insights</strong> - Discover patterns and recommendations</li>
            <li>💾 <strong>Download cleaned data</strong> - Get your processed dataset</li>
        </ul>
        <p><em>Simply upload a CSV or Excel file to begin your analysis journey!</em></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Made with ❤️ using Streamlit | Data Analyzer Pro v2.0"
    "</div>", 
    unsafe_allow_html=True
)