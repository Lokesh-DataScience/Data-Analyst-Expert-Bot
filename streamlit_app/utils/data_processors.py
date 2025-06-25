import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Any, Optional
import re

class DataProcessor:
    """Handles data processing and statistical analysis"""
    
    @staticmethod
    def parse_statistical_summary(summary_text: str) -> bool:
        """Parse statistical summary text into structured data"""
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
                    
                    # Handle multi-word variable names
                    if len(parts) > 2 and not DataProcessor._is_numeric_string(parts[1]):
                        potential_values = parts[2:]
                        if potential_values and any(DataProcessor._is_numeric_string(v) for v in potential_values[:3]):
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
            
            # Process numeric variables
            numeric_vars = []
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
            
            if numeric_vars:
                DataProcessor._display_numeric_summary_table(numeric_vars)
            
            return True
            
        except Exception as e:
            st.error(f"Error parsing statistical summary: {e}")
            return False
    
    @staticmethod
    def _is_numeric_string(s: str) -> bool:
        """Check if string represents a number"""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def _display_numeric_summary_table(numeric_data: List[Dict]):
        """Display formatted numeric summary table"""
        try:
            df = pd.DataFrame(numeric_data)
            df = df.set_index('Variable')
            
            # Format numeric columns
            numeric_columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Skewness', 'Kurtosis']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove all-NaN columns
            df = df.dropna(axis=1, how='all')
            
            # Style the dataframe
            styled_df = df.style.format({
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
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Display summary metrics
            DataProcessor._display_summary_metrics(df)
            
        except Exception as e:
            st.error(f"Error displaying numeric summary: {e}")
            st.dataframe(pd.DataFrame(numeric_data), use_container_width=True)
    
    @staticmethod
    def _display_summary_metrics(df: pd.DataFrame):
        """Display summary metrics"""
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
        
        with col3:
            if 'Std' in df.columns:
                avg_std = df['Std'].mean()
                st.metric(
                    label="📊 Average Std Dev",
                    value=f"{avg_std:.2f}",
                    help="Average standard deviation"
                )
        
        with col4:
            if 'Count' in df.columns:
                total_obs = df['Count'].sum()
                st.metric(
                    label="🔢 Total Observations",
                    value=f"{total_obs:.0f}",
                    help="Sum of all observations"
                )
    
    @staticmethod
    def parse_correlation_matrix(corr_text: str) -> Optional[pd.DataFrame]:
        """Parse correlation matrix text into DataFrame"""
        try:
            lines = [line.strip() for line in corr_text.strip().split('\n') if line.strip()]
            
            if not lines:
                return None
            
            # Remove header lines
            data_lines = []
            for line in lines:
                if not re.search(r'-?\d+\.?\d*', line):
                    continue
                data_lines.append(line)
            
            if not data_lines:
                return None
            
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
                    except ValueError:
                        continue
                
                if numeric_values:
                    variable_names.append(var_name)
                    correlation_data.append(numeric_values)
            
            if not correlation_data:
                return None
            
            # Make matrix square
            max_cols = max(len(row) for row in correlation_data)
            for row in correlation_data:
                while len(row) < max_cols:
                    row.append(np.nan)
            
            # Create DataFrame
            col_names = variable_names[:max_cols] if len(variable_names) >= max_cols else \
                       variable_names + [f'Var_{i+1}' for i in range(len(variable_names), max_cols)]
            
            df = pd.DataFrame(correlation_data, index=variable_names, columns=col_names)
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            
            # Make square
            if df.shape[0] != df.shape[1]:
                min_dim = min(df.shape[0], df.shape[1])
                df = df.iloc[:min_dim, :min_dim]
            
            return df
            
        except Exception as e:
            st.error(f"Error parsing correlation matrix: {e}")
            return None
