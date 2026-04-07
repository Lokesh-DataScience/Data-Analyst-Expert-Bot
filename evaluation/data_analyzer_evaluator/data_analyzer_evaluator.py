import pandas as pd
import numpy as np
import utils.data_analyzer as dataanalyzer

class DataAnalyzerEvaluator:
    def __init__(self):
        self.analyzer = dataanalyzer.DataAnalyzer()

    # -------------------------------
    # 1. CLEANING EVALUATION
    # -------------------------------
    def evaluate_cleaning(self, raw_df, cleaned_df):
        metrics = {}

        # Missing values reduction
        raw_missing = raw_df.isnull().sum().sum()
        cleaned_missing = cleaned_df.isnull().sum().sum()

        metrics["missing_reduction"] = (
            (raw_missing - cleaned_missing) / max(1, raw_missing)
        )

        # Duplicate removal
        raw_dup = raw_df.duplicated().sum()
        cleaned_dup = cleaned_df.duplicated().sum()

        metrics["duplicate_removal"] = 1 if cleaned_dup == 0 else 0

        # Column reduction quality
        metrics["column_reduction"] = (
            1 - (cleaned_df.shape[1] / raw_df.shape[1])
        )

        # Data type improvement
        metrics["numeric_columns"] = len(cleaned_df.select_dtypes(include=[np.number]).columns)

        return metrics

    # -------------------------------
    # 2. STATISTICAL VALIDITY
    # -------------------------------
    def evaluate_statistics(self, df, stats_output):
        metrics = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Check if key stats exist
        metrics["has_mean"] = int("mean" in stats_output.lower())
        metrics["has_skew"] = int("skew" in stats_output.lower())
        metrics["has_kurtosis"] = int("kurtosis" in stats_output.lower())
        metrics["has_correlation"] = int("correlation" in stats_output.lower())

        # Coverage score
        metrics["stat_coverage"] = np.mean(list(metrics.values()))

        return metrics

    # -------------------------------
    # 3. VISUALIZATION COVERAGE
    # -------------------------------
    def evaluate_visuals(self, plots):
        metrics = {}

        expected_keys = [
            "overview",
            "correlation",
            "distributions",
            "boxplots",
            "categorical"
        ]

        coverage = sum(1 for k in expected_keys if k in plots)

        metrics["visual_coverage"] = coverage / len(expected_keys)
        metrics["num_plots"] = len(plots)

        return metrics

    # -------------------------------
    # FULL PIPELINE
    # -------------------------------
    def evaluate(self, df):
        cleaned_df, log = self.analyzer.deep_clean_data(df)

        stats = self.analyzer.statistical_analysis(cleaned_df)

        plots = self.analyzer.create_visualizations(cleaned_df)

        return {
            "cleaning": self.evaluate_cleaning(df, cleaned_df),
            "statistics": self.evaluate_statistics(cleaned_df, stats),
            "visuals": self.evaluate_visuals(plots)
        }