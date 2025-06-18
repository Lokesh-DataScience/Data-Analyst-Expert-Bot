import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import io
import base64
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI-Powered Data Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .insight-box {
        background: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid var(--primary-color);
        margin: 1rem 0;
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

class DataAnalyzer:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
    def deep_clean_data(self, df):
        """Advanced data cleaning: string, date, categorical, outliers, irrelevant columns, encoding"""
        cleaned_df = df.copy()
        cleaning_log = []
        initial_shape = cleaned_df.shape
        cleaning_log.append(f"Initial dataset shape: {initial_shape}")

        # 1. Strip whitespace and normalize strings
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower().str.replace(r'[^\w\s]', '', regex=True)
        cleaning_log.append("Normalized string columns (trim, lower, remove special chars)")

        # 2. Parse dates
        date_cols = []
        for col in cleaned_df.columns:
            try:
                parsed = pd.to_datetime(cleaned_df[col], errors='coerce')
                if parsed.notnull().sum() > 0 and parsed.notnull().sum() > 0.5 * len(parsed):
                    cleaned_df[col] = parsed
                    date_cols.append(col)
            except Exception:
                continue
        if date_cols:
            cleaning_log.append(f"Parsed date columns: {date_cols}")

        # 3. Normalize categorical values
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].replace({
                'yes': 'yes', 'y': 'yes', 'no': 'no', 'n': 'no',
                'nan': np.nan, 'none': np.nan, 'null': np.nan
            })
        cleaning_log.append("Standardized common categorical values (yes/no, nan)")

        # 4. Remove columns with >50% missing or only 1 unique value
        cols_to_drop = [col for col in cleaned_df.columns if cleaned_df[col].isnull().mean() > 0.5 or cleaned_df[col].nunique() <= 1]
        if cols_to_drop:
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)
            cleaning_log.append(f"Dropped columns with >50% missing or 1 unique value: {cols_to_drop}")

        # 5. Remove duplicate rows
        duplicates = cleaned_df.duplicated().sum()
        if duplicates > 0:
            cleaned_df = cleaned_df.drop_duplicates()
            cleaning_log.append(f"Removed {duplicates} duplicate rows")

        # 6. Handle missing values
        missing_before = cleaned_df.isnull().sum().sum()
        if missing_before > 0:
            # Numeric: median, Categorical: mode
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype in [np.float64, np.int64]:
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                elif cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'unknown'
                    cleaned_df[col].fillna(mode_val, inplace=True)
                elif np.issubdtype(cleaned_df[col].dtype, np.datetime64):
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
            cleaning_log.append(f"Handled {missing_before} missing values")

        # 7. Advanced outlier handling (IQR)
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        outliers_capped = 0
        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask = (cleaned_df[col] < lower) | (cleaned_df[col] > upper)
            outliers_capped += mask.sum()
            cleaned_df[col] = cleaned_df[col].clip(lower, upper)
        if outliers_capped > 0:
            cleaning_log.append(f"Capped {outliers_capped} outliers using IQR method")

        # 8. Encode categorical columns
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            if cleaned_df[col].nunique() < 20:
                cleaned_df[col] = cleaned_df[col].astype('category').cat.codes
        cleaning_log.append("Encoded categorical columns with <20 unique values")

        final_shape = cleaned_df.shape
        cleaning_log.append(f"Final dataset shape: {final_shape}")
        return cleaned_df, cleaning_log
    
    def generate_insights(self, df, cleaning_log):
        """Generate AI-powered insights using Gemini, remove markdown, and provide more actionable output"""
        summary = f"""
        Dataset Overview:
        - Shape: {df.shape}
        - Columns: {list(df.columns)}
        - Data types: {df.dtypes.to_dict()}
        - Missing values: {df.isnull().sum().to_dict()}
        - Numeric summary: {df.describe().to_dict()}
        - Categorical summary: {[{col: df[col].value_counts().to_dict()} for col in df.select_dtypes(include=['object','category']).columns]}
        Cleaning Operations Performed:
        {chr(10).join(cleaning_log)}
        Sample data:
        {df.head().to_string()}
        """
        prompt = f"""
        Analyze this dataset and provide deep, actionable insights. Do not use markdown or asterisks. Format your response in clear sections with numbered or bulleted lists. Include:
        1. Key findings and patterns
        2. Data quality assessment
        3. Potential business or research insights
        4. Recommendations for further analysis
        5. Notable correlations or trends
        6. Statistical anomalies or outliers
        7. Any detected data issues or suggestions
        8. If possible, suggest predictive features or targets
        {summary}
        """
        try:
            response = self.model.generate_content(prompt)
            # Remove markdown/asterisks if any
            text = response.text.replace('*', '').replace('**', '')
            return text
        except Exception as e:
            error_msg = str(e)
            if '429' in error_msg or 'quota' in error_msg.lower():
                return ("<span style='color:red;'><b>⚠️ Gemini API quota exceeded.</b><br>"
                        "You have reached your usage limit for the Gemini API.<br>"
                        "Please check your <a href='https://ai.google.dev/gemini-api/docs/rate-limits' target='_blank'>quota and billing details</a>.<br>"
                        "Wait a minute and try again, or upgrade your plan if needed.</span>")
            return f"Error generating insights: {error_msg}"
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations including advanced plots"""
        plots = {}
        
        # 1. Data Overview
        fig_overview = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Dataset Shape', 'Data Types', 'Missing Values', 'Memory Usage'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Dataset shape
        fig_overview.add_trace(
            go.Indicator(
                mode="number",
                value=df.shape[0] * df.shape[1],
                title={"text": f"<br>({df.shape[0]} rows × {df.shape[1]} cols)"},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )
        
        # Data types
        dtype_counts = df.dtypes.value_counts()
        fig_overview.add_trace(
            go.Bar(x=dtype_counts.index.astype(str), y=dtype_counts.values, name="Data Types"),
            row=1, col=2
        )
        
        # Missing values
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if len(missing_counts) > 0:
            fig_overview.add_trace(
                go.Bar(x=missing_counts.index, y=missing_counts.values, name="Missing Values"),
                row=2, col=1
            )
        
        fig_overview.update_layout(height=600, title_text="Dataset Overview")
        plots['overview'] = fig_overview
        
        # 2. Correlation Heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            correlation_matrix = numeric_df.corr()
            fig_corr = px.imshow(
                correlation_matrix,
                title="Correlation Heatmap",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            fig_corr.update_layout(height=600)
            plots['correlation'] = fig_corr
        
        # 3. Distribution plots
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig_dist = make_subplots(
                rows=min(3, len(numeric_cols)), 
                cols=min(2, len(numeric_cols)),
                subplot_titles=[f"{col} Distribution" for col in numeric_cols[:6]]
            )
            
            for i, col in enumerate(numeric_cols[:6]):
                row = i // 2 + 1
                col_pos = i % 2 + 1
                fig_dist.add_trace(
                    go.Histogram(x=df[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig_dist.update_layout(height=800, title_text="Distribution Plots")
            plots['distributions'] = fig_dist
        
        # 4. Box plots for outlier detection
        if len(numeric_cols) > 0:
            fig_box = go.Figure()
            for col in numeric_cols[:5]:  # Limit to first 5 columns
                fig_box.add_trace(go.Box(y=df[col], name=col))
            fig_box.update_layout(title="Box Plots - Outlier Detection", height=500)
            plots['boxplots'] = fig_box
        
        # 5. Categorical analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            fig_cat = make_subplots(
                rows=min(2, len(categorical_cols)), 
                cols=min(2, len(categorical_cols)),
                subplot_titles=[f"{col} Distribution" for col in categorical_cols[:4]]
            )
            
            for i, col in enumerate(categorical_cols[:4]):
                value_counts = df[col].value_counts().head(10)
                row = i // 2 + 1
                col_pos = i % 2 + 1
                fig_cat.add_trace(
                    go.Bar(x=value_counts.index, y=value_counts.values, name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig_cat.update_layout(height=600, title_text="Categorical Variables Distribution")
            plots['categorical'] = fig_cat
        
        # 6. Pairplot (scatter matrix)
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            fig_pair = px.scatter_matrix(numeric_df, title="Pairplot (Scatter Matrix)")
            fig_pair.update_layout(height=800)
            plots['pairplot'] = fig_pair
        
        # 7. Violin plots
        if len(numeric_df.columns) > 0:
            fig_violin = go.Figure()
            for col in numeric_df.columns[:5]:
                fig_violin.add_trace(go.Violin(y=df[col], name=col, box_visible=True, meanline_visible=True))
            fig_violin.update_layout(title="Violin Plots", height=500)
            plots['violin'] = fig_violin
        
        # 8. Pie charts for categorical columns
        categorical_cols = df.select_dtypes(include=['object','category']).columns
        for col in categorical_cols[:2]:
            value_counts = df[col].value_counts().head(6)
            fig_pie = px.pie(values=value_counts.values, names=value_counts.index, title=f"{col} Proportion")
            plots[f'pie_{col}'] = fig_pie
        
        # 9. Missing value heatmap
        if df.isnull().sum().sum() > 0:
            fig_missing = px.imshow(df.isnull(), aspect='auto', color_continuous_scale='Blues', title="Missing Value Heatmap")
            plots['missing_heatmap'] = fig_missing
        
        return plots

    def statistical_analysis(self, df):
        """Return a deep statistical summary including normality, skew, kurtosis, and correlations"""
        stats_report = []
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            desc = numeric_df.describe().T
            desc['skew'] = numeric_df.skew()
            desc['kurtosis'] = numeric_df.kurtosis()
            stats_report.append("Numeric Variable Summary:")
            stats_report.append(desc.to_string())
            # Correlation significance
            corr = numeric_df.corr()
            stats_report.append("Correlation Matrix:")
            stats_report.append(corr.to_string())
        categorical_df = df.select_dtypes(include=['object','category'])
        if not categorical_df.empty:
            stats_report.append("Categorical Variable Summary:")
            for col in categorical_df.columns:
                stats_report.append(f"{col}: {categorical_df[col].nunique()} unique values. Top: {categorical_df[col].value_counts().head().to_dict()}")
        return '\n'.join(stats_report)

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Powered Data Analysis Platform</h1>
        <p>Upload your dataset and let AI handle the complete analysis pipeline</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for API key
    with st.sidebar:
        st.header("🔑 Configuration")
        api_key = st.text_input("Enter Gemini API Key", type="password", help="Get your API key from Google AI Studio")
        
        if api_key:
            st.success("API Key configured!")
        else:
            st.warning("Please enter your Gemini API key to proceed")
    
    # Main content
    if api_key:
        analyzer = DataAnalyzer(api_key)
        
        # File upload
        st.header("📁 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel (.xlsx, .xls)"
        )
        
        if uploaded_file is not None:
            try:
                # Read file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Display raw data
                with st.expander("📊 Raw Data Preview"):
                    st.dataframe(df.head(10))
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", df.shape[0])
                    with col2:
                        st.metric("Columns", df.shape[1])
                    with col3:
                        st.metric("Missing Values", df.isnull().sum().sum())
                
                # Data cleaning
                st.header("🧹 Data Cleaning")
                if st.button("🚀 Start Cleaning & Analysis", type="primary"):
                    with st.spinner("Deep cleaning data..."):
                        cleaned_df, cleaning_log = analyzer.deep_clean_data(df)
                    
                    st.success("Data cleaning completed!")
                    
                    # Display cleaning log
                    with st.expander("📋 Cleaning Log"):
                        for log in cleaning_log:
                            st.write(f"• {log}")
                    
                    # Display cleaned data
                    with st.expander("✨ Cleaned Data Preview"):
                        st.dataframe(cleaned_df.head(10))
                    
                    # Download cleaned data
                    csv = cleaned_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Cleaned Dataset",
                        data=csv,
                        file_name="cleaned_dataset.csv",
                        mime="text/csv"
                    )
                    
                    # Visualizations
                    st.header("📈 Data Visualizations")
                    with st.spinner("Creating visualizations..."):
                        plots = analyzer.create_visualizations(cleaned_df)
                    
                    # Display plots
                    for plot_name, fig in plots.items():
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # AI Insights
                    st.header("🤖 AI-Generated Insights")
                    with st.spinner("Generating insights with Gemini AI..."):
                        insights = analyzer.generate_insights(cleaned_df, cleaning_log)
                    
                    st.markdown(f"""
                    <div class="insight-box">
                        <h3>📊 Data Analysis Report</h3>
                        {insights.replace(chr(10), '<br>') if 'quota exceeded' not in insights else insights}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Statistical Summary
                    st.header("📊 Statistical Summary")
                    with st.spinner("Running deep statistical analysis..."):
                        stats_report = analyzer.statistical_analysis(cleaned_df)
                    st.code(stats_report)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("👆 Please enter your Gemini API key in the sidebar to get started")
        st.markdown("""
        ### How to get your Gemini API Key:
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Create a new API key
        3. Copy and paste it in the sidebar
        
        ### What this app does:
        - 🧹 **Automatic Data Cleaning**: Handles missing values, duplicates, outliers
        - 📊 **Comprehensive Visualizations**: Correlation heatmaps, distributions, box plots
        - 🤖 **AI-Powered Insights**: Uses Gemini 2.0 Flash for intelligent analysis
        - 💾 **Export Results**: Download cleaned dataset and analysis reports
        """)

if __name__ == "__main__":
    main()