import pandas as pd
import numpy as np
from typing import Dict, Any, List

def calculate_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates a comprehensive Data Quality Score out of 100 based on missing values,
    duplicates, outliers, data type consistency, and memory usage.
    """
    if df is None or df.empty:
        return {"score": 0, "missing": 0, "duplicates": 0, "outliers": 0, "consistency": 0, "memory": 0}

    total_cells = df.size
    total_rows = len(df)
    total_cols = len(df.columns)

    # 1. Missing Values Score (25%)
    missing_cells = df.isnull().sum().sum()
    missing_pct = missing_cells / total_cells if total_cells > 0 else 0
    missing_score = max(0, 100 - (missing_pct * 100))

    # 2. Duplicate Rows Score (20%)
    duplicate_rows = df.duplicated().sum()
    dup_pct = duplicate_rows / total_rows if total_rows > 0 else 0
    duplicate_score = max(0, 100 - (dup_pct * 100))

    # 3. Outlier Score (20%)
    # Simple IQR outlier count on numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_cells = 0
    total_numeric_cells = 0
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = col_data[(col_data < lower) | (col_data > upper)]
            outlier_cells += len(outliers)
            total_numeric_cells += len(col_data)

    outlier_pct = outlier_cells / total_numeric_cells if total_numeric_cells > 0 else 0
    outlier_score = max(0, 100 - (outlier_pct * 150)) # Outliers penalized slightly more heavily

    # 4. Consistency / Mixed Types Score (15%)
    # Check if object columns contain a mixture of string, numeric, list, etc.
    mixed_cols = 0
    object_cols = df.select_dtypes(include=['object', 'str']).columns
    
    for col in object_cols:
        types = df[col].dropna().apply(lambda x: type(x).__name__).nunique()
        if types > 1:
            mixed_cols += 1

    consistency_score = 100
    if len(object_cols) > 0:
        consistency_score = max(0, 100 - (mixed_cols / len(object_cols) * 100))

    # 5. Memory Efficiency Score (20%)
    # Penalty for large float64/int64 if smaller sizes are sufficient, 
    # or high cardinality objects that could be category type.
    memory_score = 100
    sub_penalties = 0
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        if 'object' in col_type:
            # If high card but still reusable, suggest category
            unique_count = df[col].nunique()
            if 0 < unique_count < 100 and unique_count < (total_rows * 0.3):
                sub_penalties += 10
        elif 'float64' in col_type:
            # Check if float32 or float16 could suffice
            max_val = df[col].max()
            if not pd.isna(max_val) and max_val < 32000:
                sub_penalties += 5
        elif 'int64' in col_type:
            max_val = df[col].max()
            min_val = df[col].min()
            if not pd.isna(max_val) and not pd.isna(min_val):
                if min_val > -32000 and max_val < 32000:
                    sub_penalties += 5

    memory_score = max(50, 100 - sub_penalties)

    # Weighted Average Score
    final_score = (
        (missing_score * 0.25) +
        (duplicate_score * 0.20) +
        (outlier_score * 0.20) +
        (consistency_score * 0.15) +
        (memory_score * 0.20)
    )

    return {
        "score": round(final_score, 1),
        "missing": round(missing_score, 1),
        "duplicates": round(duplicate_score, 1),
        "outliers": round(outlier_score, 1),
        "consistency": round(consistency_score, 1),
        "memory": round(memory_score, 1)
    }

def get_improvement_suggestions(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Analyzes the dataframe and returns structured suggestions for improving data quality.
    """
    suggestions = []
    if df is None or df.empty:
        return suggestions

    total_rows = len(df)

    # Missing Value suggestions
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            pct = (count / total_rows) * 100
            severity = "danger" if pct > 40 else "warning"
            suggestions.append({
                "category": "Missing Values",
                "message": f"Column **{col}** has {count} missing values ({pct:.1f}%).",
                "action": "Use Impute or Drop under Data Cleaning.",
                "severity": severity
            })

    # Duplicate Row suggestions
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        pct = (duplicates / total_rows) * 100
        suggestions.append({
            "category": "Duplicates",
            "message": f"The dataset contains {duplicates} duplicate rows ({pct:.1f}%).",
            "action": "Remove duplicate rows under Data Cleaning.",
            "severity": "warning"
        })

    # Empty/Constant Column suggestions
    for col in df.columns:
        # Check empty
        if df[col].isnull().all():
            suggestions.append({
                "category": "Column Check",
                "message": f"Column **{col}** is completely empty.",
                "action": "Drop this column to clean up your workspace.",
                "severity": "danger"
            })
        # Check constant
        elif df[col].nunique() == 1:
            suggestions.append({
                "category": "Column Check",
                "message": f"Column **{col}** has a single constant value: '{df[col].dropna().iloc[0]}'.",
                "action": "Remove constant columns as they offer zero predictive variance.",
                "severity": "info"
            })

    # Outlier suggestions
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = col_data[(col_data < lower) | (col_data > upper)]
            if len(outliers) > 0:
                pct = (len(outliers) / len(col_data)) * 100
                if pct > 1.0:
                    suggestions.append({
                        "category": "Outliers",
                        "message": f"Column **{col}** contains {len(outliers)} statistical outliers ({pct:.1f}%).",
                        "action": "Apply Outlier Treatment (Cap, Winsorize or Remove) in the Outliers tab.",
                        "severity": "info"
                    })

    # Mixed Type/Consistency suggestions
    object_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in object_cols:
        types = df[col].dropna().apply(lambda x: type(x).__name__).value_counts()
        if len(types) > 1:
            type_list = ", ".join(types.index)
            suggestions.append({
                "category": "Data Types",
                "message": f"Column **{col}** contains mixed types: ({type_list}).",
                "action": "Convert the column to a consistent type (e.g. String or Float) under Data Type Management.",
                "severity": "warning"
            })

    # Memory Optimization suggestions
    for col in df.columns:
        col_type = str(df[col].dtype)
        if 'object' in col_type:
            unique_count = df[col].nunique()
            if 0 < unique_count < 100 and unique_count < (total_rows * 0.3):
                suggestions.append({
                    "category": "Optimization",
                    "message": f"Column **{col}** has low cardinality object types ({unique_count} unique values).",
                    "action": "Convert it to Category type to save memory and boost analysis speed.",
                    "severity": "info"
                })

    return suggestions
