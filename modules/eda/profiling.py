import pandas as pd
import numpy as np
from typing import Dict, Any, List

def get_column_profile(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """
    Computes statistical indicators for a single column in the DataFrame.
    """
    profile = {}
    if df is None or col not in df.columns:
        return profile

    series = df[col]
    total_count = len(df)
    non_null_count = series.count()
    missing_count = series.isnull().sum()
    missing_pct = (missing_count / total_count) * 100 if total_count > 0 else 0
    unique_count = series.nunique()
    
    profile["name"] = col
    profile["dtype"] = str(series.dtype)
    profile["count"] = total_count
    profile["non_null"] = non_null_count
    profile["missing"] = missing_count
    profile["missing_pct"] = round(missing_pct, 2)
    profile["unique"] = unique_count

    # Numeric Column Profile
    if pd.api.types.is_numeric_dtype(series):
        series_clean = series.dropna()
        if len(series_clean) > 0:
            profile["mean"] = round(float(series_clean.mean()), 4)
            profile["median"] = round(float(series_clean.median()), 4)
            profile["min"] = round(float(series_clean.min()), 4)
            profile["max"] = round(float(series_clean.max()), 4)
            profile["var"] = round(float(series_clean.var()), 4)
            profile["std"] = round(float(series_clean.std()), 4)
            
            # Mode
            mode_s = series_clean.mode()
            profile["mode"] = round(float(mode_s.iloc[0]), 4) if not mode_s.empty else "N/A"
            
            # Skewness and Kurtosis
            try:
                profile["skew"] = round(float(series_clean.skew()), 4)
            except Exception:
                profile["skew"] = "N/A"
                
            try:
                profile["kurtosis"] = round(float(series_clean.kurt()), 4)
            except Exception:
                profile["kurtosis"] = "N/A"
        else:
            profile.update({
                "mean": "N/A", "median": "N/A", "mode": "N/A", "min": "N/A", "max": "N/A",
                "var": "N/A", "std": "N/A", "skew": "N/A", "kurtosis": "N/A"
            })
    else:
        # Non-numeric Column Profile
        series_clean = series.dropna().astype(str)
        if len(series_clean) > 0:
            # Mode
            mode_s = series_clean.mode()
            profile["mode"] = mode_s.iloc[0] if not mode_s.empty else "N/A"
            
            # Most frequent values (top 5)
            val_counts = series_clean.value_counts().head(5)
            profile["frequent_values"] = [{"value": k, "count": int(v), "pct": round((v/len(series_clean))*100, 1)} for k, v in val_counts.items()]
        else:
            profile["mode"] = "N/A"
            profile["frequent_values"] = []

    return profile

def get_dataset_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes global metrics for the dataset dashboard.
    """
    if df is None:
        return {}

    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Types counts
    num_cols = len(df.select_dtypes(include=[np.number]).columns)
    bool_cols = len(df.select_dtypes(include=['bool']).columns)
    dt_cols = len(df.select_dtypes(include=['datetime64']).columns)
    cat_cols = len(df.select_dtypes(include=['category']).columns)
    
    # Everything else is classified as object/text
    obj_cols = total_cols - num_cols - bool_cols - dt_cols - cat_cols

    missing_cells = df.isnull().sum().sum()
    total_cells = df.size
    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    duplicate_rows = df.duplicated().sum()
    
    # Memory size in MB
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)

    return {
        "rows": total_rows,
        "cols": total_cols,
        "numeric": num_cols,
        "categorical": cat_cols,
        "boolean": bool_cols,
        "datetime": dt_cols,
        "object": obj_cols,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_pct, 2),
        "duplicates": duplicate_rows,
        "memory_mb": round(memory_mb, 2)
    }
