import streamlit as st
import pandas as pd
import numpy as np
import re

def display_analysis_results(response):
    st.subheader("🧹 Data Cleaning Log")
    for i, step in enumerate(response.get("cleaning_log", []), 1):
        st.write(f"{i}. {step}")

    stats = response.get("statistical_summary")
    if isinstance(stats, str):
        if "Correlation Matrix:" in stats:
            summary_text, corr_text = stats.split("Correlation Matrix:")
        else:
            summary_text, corr_text = stats, ""

        if "Numeric Variable Summary:" in summary_text:
            summary_text = summary_text.replace("Numeric Variable Summary:", "").strip()
        
        st.subheader("📈 Numeric Variable Summary")
        if summary_text.strip():
            success = parse_statistical_summary(summary_text.strip())
            if not success:
                st.warning("Could not parse statistical summary. Displaying raw text:")
                st.code(summary_text)
        else:
            st.info("No numeric variable summary available")

        if corr_text.strip():
            display_correlation_matrix_improved(corr_text.strip())
                
    elif isinstance(stats, dict):
        st.subheader("📊 Statistical Summary")
        st.json(stats)

    col_info = response.get("column_info", {})
    with st.expander("📋 Column Info", expanded=False):
        if isinstance(col_info, dict):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Columns:**")
                orig_cols = col_info.get("original_columns", [])
                if orig_cols:
                    for i, col in enumerate(orig_cols, 1):
                        st.write(f"{i}. {col}")
                st.markdown("**Data Types:**")
                dt = col_info.get("data_types", {})
                if dt:
                    dtype_df = pd.DataFrame(list(dt.items()), columns=["Column", "Type"])
                    st.dataframe(dtype_df, use_container_width=True)
            with col2:
                st.markdown("**Cleaned Columns:**")
                clean_cols = col_info.get("cleaned_columns", [])
                if clean_cols:
                    for i, col in enumerate(clean_cols, 1):
                        st.write(f"{i}. {col}")
                st.markdown("**Missing Values:**")
                mv = col_info.get("missing_values", {})
                if mv:
                    missing_df = pd.DataFrame(list(mv.items()), columns=["Column", "Missing Count"])
                    st.dataframe(missing_df, use_container_width=True)
            st.markdown("**Unique Value Counts:**")
            uc = col_info.get("unique_counts", {})
            if uc:
                unique_df = pd.DataFrame(list(uc.items()), columns=["Column", "Unique Values"])
                st.dataframe(unique_df, use_container_width=True)
        else:
            st.text(col_info)

    st.subheader("✨ Cleaned Data Preview")
    cleaned_sample = response.get("sample_data", {}).get("cleaned", [])
    if cleaned_sample:
        try:
            preview_df = pd.DataFrame(cleaned_sample)
            st.dataframe(preview_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying cleaned data: {e}")
            st.json(cleaned_sample[:5])

    st.subheader("📈 Visualizations")
    plots = response.get("visualizations", {})
    if plots:
        import plotly.graph_objects as go
        for name, fig_dict in plots.items():
            if "error" in fig_dict:
                st.warning(f"{name}: {fig_dict['error']}")
            else:
                try:
                    fig = go.Figure(fig_dict)
                    fig.update_layout(
                        title=name.replace('_', ' ').title(),
                        showlegend=True,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render plot {name}: {e}")
                    st.json(fig_dict)
    else:
        st.info("No visualizations available")

    insights = response.get("insights")
    if insights:
        st.subheader("🤖 AI-Generated Insights")
        if isinstance(insights, str):
            insight_parts = insights.split('\n')
            formatted_insights = []
            for part in insight_parts:
                part = part.strip()
                if part:
                    if part.startswith('•') or part.startswith('-') or part.startswith('*'):
                        formatted_insights.append(f"• {part[1:].strip()}")
                    elif part.endswith(':'):
                        formatted_insights.append(f"**{part}**")
                    else:
                        formatted_insights.append(part)
            st.markdown('\n\n'.join(formatted_insights))
        else:
            st.write(insights)
    else:
        st.info("No AI-generated insights available")

def is_numeric_string(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def parse_statistical_summary(summary_text):
    try:
        lines = [line.strip() for line in summary_text.split('\n') if line.strip()]
        if not lines:
            return False
        parsed_rows = []
        for line in lines:
            parts = [p.strip() for p in line.split()]
            if len(parts) >= 2:
                var_name = parts[0]
                value_start_idx = 1
                if len(parts) > 2 and not is_numeric_string(parts[1]):
                    potential_values = parts[2:]
                    if potential_values and any(is_numeric_string(v) for v in potential_values[:3]):
                        var_name = f"{parts[0]} {parts[1]}"
                        value_start_idx = 2
                values = []
                for val in parts[value_start_idx:]:
                    try:
                        float_val = float(val)
                        values.append(float_val)
                    except ValueError:
                        if val.lower() in ['none', 'nan', 'n/a', '']:
                            values.append(np.nan)
                        else:
                            values.append(val)
                if values:
                    parsed_rows.append({'Variable': var_name, 'Values': values})
        if not parsed_rows:
            return False
        numeric_vars = []
        categorical_vars = []
        for row in parsed_rows:
            var_name = row['Variable']
            values = row['Values']
            if len(values) >= 6 and all(isinstance(v, (int, float)) or pd.isna(v) for v in values[:8]):
                stat_dict = {
                    'Variable': var_name,
                    'Count': values[0] if len(values) > 0 else np.nan,
                    'Mean': values[1] if len(values) > 1 else np.nan,
                    'Std': values[2] if len(values) > 2 else np.nan,
                    'Min': values[3] if len(values) > 3 else np.nan,
                    '25%': values[4] if len(values) > 4 else np.nan,
                    '50%': values[5] if len(values) > 5 else np.nan,
                    '75%': values[6] if len(values) > 6 else np.nan,
                    'Max': values[7] if len(values) > 7 else np.nan
                }
                if len(values) > 8:
                    stat_dict['Skewness'] = values[8]
                if len(values) > 9:
                    stat_dict['Kurtosis'] = values[9]
                numeric_vars.append(stat_dict)
            else:
                categorical_vars.append({
                    'Variable': var_name,
                    'Info': ' '.join([str(v) for v in values])
                })
        if numeric_vars:
            display_numeric_summary_table(numeric_vars)
        return True
    except Exception as e:
        st.error(f"Error parsing statistical summary: {e}")
        return False

def display_numeric_summary_table(numeric_data):
    try:
        df = pd.DataFrame(numeric_data)
        df = df.set_index('Variable')
        numeric_columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Skewness', 'Kurtosis']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(axis=1, how='all')
        formatted_df = df.style.format({
            'Count': '{:.0f}',
            'Mean': '{:.4f}',
            'Std': '{:.4f}',
            'Min': '{:.2f}',
            '25%': '{:.2f}',
            '50%': '{:.2f}',
            '75%': '{:.2f}',
            'Max': '{:.2f}',
            'Skewness': '{:.4f}',
            'Kurtosis': '{:.4f}'
        }).background_gradient(
            cmap='RdYlBu_r',
            subset=['Mean', 'Std'] if 'Mean' in df.columns and 'Std' in df.columns else None
        ).set_properties(**{
            'text-align': 'center',
            'font-size': '12px',
            'border': '1px solid #ddd',
            'padding': '8px'
        }).set_table_styles([
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#2c3e50'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('padding', '12px'),
                    ('border', '1px solid #34495e')
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('text-align', 'center'),
                    ('padding', '8px'),
                    ('border', '1px solid #ddd')
                ]
            },
            {
                'selector': '.index_name',
                'props': [
                    ('font-weight', 'bold'),
                    ('background-color', '#ecf0f1'),
                    ('text-align', 'left'),
                    ('padding-left', '12px')
                ]
            },
            {
                'selector': 'table',
                'props': [
                    ('border-collapse', 'collapse'),
                    ('margin', '10px auto'),
                    ('width', '100%'),
                    ('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
                ]
            }
        ])
        st.dataframe(formatted_df, use_container_width=True)
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="📊 Variables",
                value=len(df),
                help="Number of numeric variables analyzed"
            )
        with col2:
            if 'Mean' in df.columns:
                avg_mean = df['Mean'].mean()
                st.metric(
                    label="📈 Average Mean",
                    value=f"{avg_mean:.2f}",
                    help="Average of all variable means"
                )
            else:
                st.metric(label="📈 Statistics", value=len(df.columns))
        with col3:
            if 'Std' in df.columns:
                avg_std = df['Std'].mean()
                st.metric(
                    label="📊 Average Std Dev",
                    value=f"{avg_std:.2f}",
                    help="Average standard deviation"
                )
            else:
                st.metric(label="📊 Numeric Cols", value=len(df.columns))
        with col4:
            if 'Count' in df.columns:
                total_obs = df['Count'].sum()
                st.metric(
                    label="🔢 Total Observations",
                    value=f"{total_obs:.0f}",
                    help="Sum of all observations"
                )
            else:
                st.metric(label="🔢 Data Points", value="N/A")
        with st.expander("📖 Understanding the Statistics"):
            st.markdown("""
            **Statistical Measures Explained:**
            
            | Statistic | Description | Interpretation |
            |-----------|-------------|----------------|
            | **Count** | Number of non-null observations | Higher = more complete data |
            | **Mean** | Average value | Central tendency of the data |
            | **Std** | Standard deviation | Higher = more variability |
            | **Min/Max** | Minimum and maximum values | Data range |
            | **25%, 50%, 75%** | Quartiles (Q1, Median, Q3) | Distribution shape |
            | **Skewness** | Measure of asymmetry | 0=symmetric, >0=right-skewed, <0=left-skewed |
            | **Kurtosis** | Measure of tail heaviness | 3=normal, >3=heavy tails, <3=light tails |
            
            **Quick Rules:**
            - **High variability**: Std > 0.5 × Mean
            - **Approximately symmetric**: |Skewness| < 0.5
            - **Normal-like distribution**: Kurtosis ≈ 3
            """)
    except Exception as e:
        st.error(f"Error displaying numeric summary: {e}")
        st.dataframe(pd.DataFrame(numeric_data), use_container_width=True)

def display_summary_section(summary_text):
    if "Numeric Variable Summary:" in summary_text:
        summary_text = summary_text.replace("Numeric Variable Summary:", "").strip()
    st.subheader("📈 Statistical Analysis Summary")
    if summary_text.strip():
        success = parse_statistical_summary(summary_text.strip())
        if not success:
            st.warning("Could not parse statistical summary. Displaying raw text:")
            with st.expander("Raw Statistical Data"):
                st.code(summary_text, language='text')
    else:
        st.info("No statistical summary available")

def display_correlation_matrix_improved(corr_text):
    st.subheader("🔗 Correlation Matrix")
    try:
        lines = [line.strip() for line in corr_text.strip().split('\n') if line.strip()]
        if not lines:
            st.warning("No correlation matrix data found")
            return
        data_lines = []
        for line in lines:
            if not re.search(r'-?\d+\.?\d*', line):
                continue
            data_lines.append(line)
        if not data_lines:
            st.warning("No numeric correlation data found")
            st.code(corr_text)
            return
        success = parse_correlation_matrix_robust(data_lines)
        if not success:
            st.info("Displaying correlation matrix as formatted text:")
            display_formatted_correlation_text(data_lines)
    except Exception as e:
        st.error(f"Error processing correlation matrix: {str(e)}")
        st.code(corr_text)
    display_correlation_help()

def parse_correlation_matrix_robust(data_lines):
    try:
        correlation_data = []
        variable_names = []
        for line in data_lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            var_name = parts[0]
            numeric_values = []
            for part in parts[1:]:
                try:
                    val = float(part)
                    if -1.1 <= val <= 1.1:
                        numeric_values.append(val)
                    else:
                        continue
                except ValueError:
                    continue
            if numeric_values:
                variable_names.append(var_name)
                correlation_data.append(numeric_values)
        if not correlation_data:
            return False
        max_cols = max(len(row) for row in correlation_data)
        min_cols = min(len(row) for row in correlation_data)
        if max_cols != min_cols:
            for row in correlation_data:
                while len(row) < max_cols:
                    row.append(np.nan)
        if len(variable_names) == max_cols:
            col_names = variable_names.copy()
        else:
            col_names = variable_names[:max_cols] if len(variable_names) >= max_cols else variable_names + [f'Var_{i+1}' for i in range(len(variable_names), max_cols)]
        col_names = make_names_unique(col_names)
        variable_names = make_names_unique(variable_names)
        df = pd.DataFrame(correlation_data, index=variable_names, columns=col_names)
        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)
        if df.shape[0] != df.shape[1]:
            min_dim = min(df.shape[0], df.shape[1])
            df = df.iloc[:min_dim, :min_dim]
        if df.empty:
            return False
        display_correlation_heatmap_fixed(df)
        return True
    except Exception as e:
        st.error(f"Error in robust parsing: {e}")
        return False

def make_names_unique(names):
    unique_names = []
    name_counts = {}
    for name in names:
        if name in name_counts:
            name_counts[name] += 1
            unique_names.append(f"{name}_{name_counts[name]}")
        else:
            name_counts[name] = 0
            unique_names.append(name)
    return unique_names

def display_correlation_heatmap_fixed(df):
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.clip(-1, 1)
        styled_df = numeric_df.style.format("{:.3f}").background_gradient(
            cmap='RdYlBu_r',
            vmin=-1, 
            vmax=1
        ).set_properties(**{
            'text-align': 'center',
            'font-weight': 'bold',
            'border': '1px solid #ddd',
            'font-size': '12px',
            'padding': '8px'
        }).set_table_styles([
            {
                'selector': 'th',
                'props': [
                    ('text-align', 'center'), 
                    ('font-weight', 'bold'),
                    ('background-color', '#2c3e50'),
                    ('color', 'white'),
                    ('font-size', '12px'),
                    ('padding', '10px'),
                    ('border', '1px solid #34495e')
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('text-align', 'center'),
                    ('font-size', '12px'),
                    ('padding', '8px')
                ]
            },
            {
                'selector': '.index_name',
                'props': [
                    ('font-weight', 'bold'),
                    ('background-color', '#ecf0f1'),
                    ('color', '#2c3e50')
                ]
            },
            {
                'selector': 'table',
                'props': [
                    ('border-collapse', 'collapse'),
                    ('margin', '10px auto'),
                    ('width', '100%'),
                    ('box-shadow', '0 4px 6px rgba(0,0,0,0.1)')
                ]
            }
        ])
        st.dataframe(styled_df, use_container_width=True)
        display_correlation_stats(numeric_df)
    except Exception as e:
        st.error(f"Error styling correlation matrix: {e}")
        st.dataframe(df, use_container_width=True)

def display_correlation_stats(df):
    try:
        mask = np.triu(np.ones_like(df, dtype=bool), k=1)
        correlations = df.where(mask).stack().dropna()
        if correlations.empty:
            correlations = df.stack().dropna()
            correlations = correlations[correlations != 1.0]
        if not correlations.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    label="🔴 Strongest Positive",
                    value=f"{correlations.max():.3f}",
                    help="Highest positive correlation in the matrix"
                )
            with col2:
                st.metric(
                    label="🔵 Strongest Negative", 
                    value=f"{correlations.min():.3f}",
                    help="Strongest negative correlation in the matrix"
                )
            with col3:
                st.metric(
                    label="📊 Average Correlation",
                    value=f"{correlations.mean():.3f}",
                    help="Mean of all correlation coefficients"
                )
            with col4:
                strong_correlations = correlations[abs(correlations) > 0.7].count()
                st.metric(
                    label="💪 Strong Correlations",
                    value=str(strong_correlations),
                    help="Number of correlations > |0.7|"
                )
    except Exception as e:
        st.warning(f"Could not calculate correlation statistics: {e}")

def display_formatted_correlation_text(data_lines):
    try:
        all_parts = [line.split() for line in data_lines]
        if not all_parts:
            st.code('\n'.join(data_lines))
            return
        max_cols = max(len(parts) for parts in all_parts)
        col_widths = [0] * max_cols
        for parts in all_parts:
            for i, part in enumerate(parts):
                if i < max_cols:
                    col_widths[i] = max(col_widths[i], len(str(part)))
        formatted_lines = []
        for parts in all_parts:
            formatted_parts = []
            for i in range(max_cols):
                if i < len(parts):
                    part = str(parts[i])
                    if i == 0:
                        formatted_parts.append(f"{part:<{col_widths[i]}}")
                    else:
                        formatted_parts.append(f"{part:>{col_widths[i]}}")
                else:
                    formatted_parts.append(" " * col_widths[i])
            formatted_lines.append("  ".join(formatted_parts))
        st.code('\n'.join(formatted_lines), language='text')
    except Exception as e:
        st.error(f"Error formatting correlation text: {e}")
        st.code('\n'.join(data_lines))

def display_correlation_help():
    with st.expander("📖 How to Read This Correlation Matrix"):
        st.markdown("""
        **Correlation Values Range from -1 to +1:**
        
        | Range | Interpretation | Strength | Example |
        |-------|---------------|----------|---------|
        | **0.8 to 1.0** | Strong positive correlation | 🟢 Very Strong | Height & Weight |
        | **0.5 to 0.8** | Moderate positive correlation | 🟡 Moderate | Education & Income |
        | **0.2 to 0.5** | Weak positive correlation | 🟠 Weak | Age & Experience |
        | **-0.2 to 0.2** | No significant correlation | ⚪ None | Random variables |
        | **-0.2 to -0.5** | Weak negative correlation | 🟠 Weak | Price & Demand |
        | **-0.5 to -0.8** | Moderate negative correlation | 🟡 Moderate | Temperature & Heating Cost |
        | **-0.8 to -1.0** | Strong negative correlation | 🔴 Very Strong | Speed & Travel Time |
        
        **Key Insights:**
        - **Diagonal = 1.0**: Every variable perfectly correlates with itself
        - **Positive**: Variables move in the same direction
        - **Negative**: Variables move in opposite directions
        - **Zero**: No linear relationship (but could have non-linear relationship)
        
        **⚠️ Important Notes:**
        - Correlation ≠ Causation
        - Only measures linear relationships
        - Outliers can significantly affect correlations
        """)