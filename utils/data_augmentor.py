import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import KNNImputer
from typing import Tuple, Dict, Any, List


class DataAugmentor:
    """Handles all data augmentation logic: diagnosis, cleaning, enrichment."""

    # ------------------------------------------------------------------ #
    #  STAGE 1 — DIAGNOSIS                                                #
    # ------------------------------------------------------------------ #

    def diagnose(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Scan the dataframe and produce a full augmentation plan."""
        issues = []
        recommendations = []
        stats_summary = {}

        total_rows, total_cols = df.shape

        # --- Duplicates ---
        duplicate_count = int(df.duplicated().sum())
        stats_summary["duplicate_rows"] = duplicate_count
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate rows.")
            recommendations.append({
                "type": "deduplication",
                "description": f"Remove {duplicate_count} duplicate rows",
                "severity": "medium"
            })

        # --- Missing values ---
        missing = df.isnull().sum()
        missing_pct = (missing / total_rows * 100).round(2)
        missing_cols = missing[missing > 0]
        stats_summary["missing_values"] = missing_cols.to_dict()
        stats_summary["missing_pct"] = missing_pct[missing_pct > 0].to_dict()

        if not missing_cols.empty:
            issues.append(f"Found missing values in {len(missing_cols)} column(s).")
            for col, count in missing_cols.items():
                pct = missing_pct[col]
                severity = "high" if pct > 30 else "medium" if pct > 10 else "low"
                recommendations.append({
                    "type": "imputation",
                    "column": col,
                    "missing_count": int(count),
                    "missing_pct": float(pct),
                    "description": f"Impute {count} missing values in '{col}' ({pct}%)",
                    "severity": severity
                })

        # --- Outliers (numeric columns only) ---
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_info = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)
            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                outlier_info[col] = outlier_count
                recommendations.append({
                    "type": "outlier_treatment",
                    "column": col,
                    "outlier_count": outlier_count,
                    "description": f"Winsorize {outlier_count} outlier(s) in '{col}'",
                    "severity": "medium" if outlier_count / len(series) > 0.05 else "low"
                })
        stats_summary["outliers"] = outlier_info

        # --- Skewed distributions ---
        skew_info = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4 or (series <= 0).any():
                continue
            skewness = float(series.skew())
            if abs(skewness) > 1.0:
                skew_info[col] = round(skewness, 3)
                recommendations.append({
                    "type": "transformation",
                    "column": col,
                    "skewness": round(skewness, 3),
                    "description": f"Apply log transform to '{col}' (skewness: {skewness:.2f})",
                    "severity": "low"
                })
        stats_summary["skewed_columns"] = skew_info

        # --- Class imbalance (categorical columns) ---
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        imbalance_info = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts(normalize=True)
            if len(value_counts) >= 2:
                imbalance_ratio = float(value_counts.iloc[0] / value_counts.iloc[-1])
                if imbalance_ratio > 3.0:
                    imbalance_info[col] = round(imbalance_ratio, 2)
                    recommendations.append({
                        "type": "class_imbalance",
                        "column": col,
                        "ratio": round(imbalance_ratio, 2),
                        "description": f"Class imbalance in '{col}' (ratio: {imbalance_ratio:.1f}:1)",
                        "severity": "medium" if imbalance_ratio > 5 else "low"
                    })
        stats_summary["class_imbalance"] = imbalance_info

        # --- Low row count ---
        stats_summary["row_count"] = total_rows
        if total_rows < 100:
            issues.append(f"Dataset has only {total_rows} rows — consider synthetic row generation.")
            recommendations.append({
                "type": "synthetic_rows",
                "description": f"Generate synthetic rows to expand dataset beyond {total_rows} rows",
                "severity": "high" if total_rows < 30 else "medium"
            })

        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "issues": issues,
            "recommendations": recommendations,
            "stats_summary": stats_summary,
            "has_issues": len(issues) > 0
        }

    # ------------------------------------------------------------------ #
    #  STAGE 2 — AUGMENTATION                                             #
    # ------------------------------------------------------------------ #

    def augment(
        self,
        df: pd.DataFrame,
        options: Dict[str, bool]
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Apply selected augmentation steps. Returns augmented df + change log."""
        augmented = df.copy()
        change_log = []

        if options.get("apply_deduplication"):
            augmented, log = self._deduplicate(augmented)
            change_log.extend(log)

        if options.get("apply_imputation"):
            augmented, log = self._impute(augmented)
            change_log.extend(log)

        if options.get("apply_outlier_treatment"):
            augmented, log = self._treat_outliers(augmented)
            change_log.extend(log)

        if options.get("apply_transformations"):
            augmented, log = self._transform_skewed(augmented)
            change_log.extend(log)

        if options.get("apply_synthetic_rows"):
            augmented, log = self._generate_synthetic_rows(augmented)
            change_log.extend(log)

        return augmented, change_log

    # ------------------------------------------------------------------ #
    #  PRIVATE HELPERS                                                     #
    # ------------------------------------------------------------------ #

    def _deduplicate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        log = []
        if removed > 0:
            log.append({
                "step": "Deduplication",
                "detail": f"Removed {removed} duplicate row(s).",
                "rows_before": before,
                "rows_after": len(df)
            })
        return df, log

    def _impute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        log = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Numeric — KNN imputation if <= 10 cols, else median
        num_missing = [c for c in numeric_cols if df[c].isnull().any()]
        if num_missing:
            if len(numeric_cols) <= 10:
                try:
                    imputer = KNNImputer(n_neighbors=5)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    log.append({
                        "step": "KNN Imputation",
                        "detail": f"Imputed missing values in: {', '.join(num_missing)} using KNN (k=5)."
                    })
                except Exception:
                    for col in num_missing:
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                    log.append({
                        "step": "Median Imputation",
                        "detail": f"Imputed missing values in: {', '.join(num_missing)} using column median."
                    })
            else:
                for col in num_missing:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                log.append({
                    "step": "Median Imputation",
                    "detail": f"Imputed missing values in: {', '.join(num_missing)} using column median."
                })

        # Categorical — mode or "Unknown"
        cat_missing = [c for c in categorical_cols if df[c].isnull().any()]
        for col in cat_missing:
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"
            df[col].fillna(fill_val, inplace=True)
            log.append({
                "step": "Categorical Imputation",
                "detail": f"Filled missing values in '{col}' with '{fill_val}'."
            })

        return df, log

    def _treat_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        log = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            Q1, Q3 = series.quantile(0.01), series.quantile(0.99)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())
            if outlier_count > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                log.append({
                    "step": "Outlier Winsorization",
                    "detail": f"Capped {outlier_count} outlier(s) in '{col}' to [{lower:.2f}, {upper:.2f}]."
                })

        return df, log

    def _transform_skewed(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        log = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4 or (series <= 0).any():
                continue
            skewness = series.skew()
            if abs(skewness) > 1.0:
                df[col] = np.log1p(df[col])
                log.append({
                    "step": "Log Transformation",
                    "detail": f"Applied log1p transform to '{col}' (original skewness: {skewness:.2f})."
                })

        return df, log

    def _generate_synthetic_rows(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        Generate synthetic rows using Gaussian noise on numeric columns
        and sampling for categorical columns. Targets 2x the original row count.
        """
        log = []
        target_rows = max(100, len(df) * 2)
        rows_to_add = target_rows - len(df)

        if rows_to_add <= 0:
            return df, log

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        synthetic_rows = []
        for _ in range(rows_to_add):
            row = {}
            for col in numeric_cols:
                col_mean = df[col].mean()
                col_std = df[col].std()
                row[col] = float(np.random.normal(col_mean, col_std * 0.1))
            for col in categorical_cols:
                row[col] = df[col].dropna().sample(1).values[0]
            synthetic_rows.append(row)

        synthetic_df = pd.DataFrame(synthetic_rows)
        augmented = pd.concat([df, synthetic_df], ignore_index=True)

        log.append({
            "step": "Synthetic Row Generation",
            "detail": f"Generated {rows_to_add} synthetic rows using Gaussian noise (numeric) and frequency sampling (categorical).",
            "rows_before": len(df),
            "rows_after": len(augmented)
        })

        return augmented, log