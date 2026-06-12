import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai

class DataAnalyzer:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("No Gemini or Google API key found in .env file.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash') 
    def deep_clean_data(self, df):
        """Advanced data cleaning: string, date, categorical, outliers, irrelevant columns, encoding"""
        cleaned_df = df.copy()
        cleaning_log = []
        initial_shape = cleaned_df.shape
        cleaning_log.append(f"Initial dataset shape: {initial_shape}")

        # 1. First, identify potential date columns before string normalization
        potential_date_cols = []
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            # Skip columns that are already datetime
            if pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                continue
                
            # Check if column might contain dates by trying to parse a sample
            sample_values = cleaned_df[col].dropna().head(10)
            if len(sample_values) > 0:
                try:
                    parsed_sample = pd.to_datetime(sample_values, format='%Y-%m-%d', errors='coerce')
                    if parsed_sample.notna().sum() > len(sample_values) * 0.7:  # 70% parseable
                        potential_date_cols.append(col)
                except:
                    continue
        
        # 2. Parse dates BEFORE string normalization
        date_cols = []
        for col in potential_date_cols:
            try:
                parsed = pd.to_datetime(cleaned_df[col], errors='coerce')
                valid_dates = parsed.notna().sum()
                if valid_dates > 0 and valid_dates > 0.3 * len(parsed):  # Lower threshold
                    cleaned_df[col] = parsed
                    date_cols.append(col)
            except Exception:
                continue
        if date_cols:
            cleaning_log.append(f"Parsed date columns: {date_cols}")

        # 3. Strip whitespace and normalize strings (EXCLUDE date columns)
        string_cols = [col for col in cleaned_df.select_dtypes(include=['object']).columns 
                    if col not in date_cols]
        
        for col in string_cols:
            # Handle NaN values before string operations
            mask = cleaned_df[col].notna()
            cleaned_df.loc[mask, col] = (cleaned_df.loc[mask, col]
                                    .astype(str)
                                    .str.strip()
                                    .str.lower()
                                    .str.replace(r'[^\w\s]', '', regex=True))
            # Convert 'nan' strings back to actual NaN
            cleaned_df[col] = cleaned_df[col].replace('nan', np.nan)
        cleaning_log.append(f"Normalized string columns (excluding dates): {string_cols}")

        # 4. Normalize categorical values (EXCLUDE date columns)
        categorical_cols = [col for col in cleaned_df.select_dtypes(include=['object']).columns 
                        if col not in date_cols]
        
        for col in categorical_cols:
            # Create a mapping for common variations
            value_mapping = {
                'yes': 'yes', 'y': 'yes', 'true': 'yes', '1': 'yes',
                'no': 'no', 'n': 'no', 'false': 'no', '0': 'no',
                'nan': np.nan, 'none': np.nan, 'null': np.nan, '': np.nan
            }
            cleaned_df[col] = cleaned_df[col].replace(value_mapping)
        cleaning_log.append("Standardized common categorical values (yes/no, nan)")

        # 4. Remove columns with >50% missing or only 1 unique value
        cols_to_drop = []
        for col in cleaned_df.columns:
            missing_pct = cleaned_df[col].isnull().mean()
            unique_count = cleaned_df[col].nunique()
            if missing_pct > 0.5 or unique_count <= 1:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)
            cleaning_log.append(f"Dropped columns with >50% missing or ≤1 unique value: {cols_to_drop}")

        # 5. Remove duplicate rows
        duplicates = cleaned_df.duplicated().sum()
        if duplicates > 0:
            cleaned_df = cleaned_df.drop_duplicates()
            cleaning_log.append(f"Removed {duplicates} duplicate rows")

        # 6. Handle missing values
        missing_before = cleaned_df.isnull().sum().sum()
        if missing_before > 0:
            for col in cleaned_df.columns:
                if cleaned_df[col].isnull().sum() == 0:  # Skip columns with no missing values
                    continue
                    
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    # Use median for numeric columns
                    median_val = cleaned_df[col].median()
                    if pd.notna(median_val):  # Only fill if median is not NaN
                        cleaned_df[col] = cleaned_df[col].fillna(median_val)  # <-- CORRECT
                elif cleaned_df[col].dtype == 'object':
                    # Use mode for categorical columns
                    mode_values = cleaned_df[col].mode()
                    mode_val = mode_values[0] if len(mode_values) > 0 else 'unknown'
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                elif pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                    # Use mode for datetime columns
                    mode_values = cleaned_df[col].mode()
                    if len(mode_values) > 0:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val)
            cleaning_log.append(f"Handled {missing_before} missing values")

        # 7. Advanced outlier handling (IQR)
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        outliers_capped = 0
        for col in numeric_cols:
            if cleaned_df[col].nunique() <= 1:  # Skip columns with no variance
                continue
                
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Only apply if there's actual variance
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                mask = (cleaned_df[col] < lower) | (cleaned_df[col] > upper)
                outliers_capped += mask.sum()
                cleaned_df[col] = cleaned_df[col].clip(lower, upper)
        
        if outliers_capped > 0:
            cleaning_log.append(f"Capped {outliers_capped} outliers using IQR method")

        # 9. Encode categorical columns (EXCLUDE date columns)
        categorical_cols_encoded = []
        non_date_object_cols = [col for col in cleaned_df.select_dtypes(include=['object']).columns 
                            if col not in date_cols]
        
        for col in non_date_object_cols:
            unique_count = cleaned_df[col].nunique()
            if unique_count < 20 and unique_count > 1:  # Only encode if there are multiple categories
                # Use label encoding (convert to category first to handle NaN properly)
                cleaned_df[col] = cleaned_df[col].astype('category')
                cleaned_df[col] = cleaned_df[col].cat.codes
                # Replace -1 (NaN category code) with NaN
                cleaned_df[col] = cleaned_df[col].replace(-1, np.nan)
                categorical_cols_encoded.append(col)
        
        if categorical_cols_encoded:
            cleaning_log.append(f"Encoded categorical columns with <20 unique values: {categorical_cols_encoded}")

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
        
        # 1. Data Overview - Only Dataset Shape and Data Types
        fig_overview = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Dataset Shape', 'Data Types'),
            specs=[[{"type": "indicator"}, {"type": "bar"}]]
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
        
        fig_overview.update_layout(height=400, title_text="Dataset Overview")
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