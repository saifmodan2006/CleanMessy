import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any

def run_data_validation(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Scans the DataFrame for standard data validation issues.
    Returns: a list of validation alerts/issues found.
    """
    issues = []
    if df is None or df.empty:
        return issues
        
    total_rows = len(df)
    
    # 1. Check for Duplicate Columns
    duplicate_cols = []
    cols = df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if df[cols[i]].equals(df[cols[j]]):
                duplicate_cols.append((cols[i], cols[j]))
                
    for col1, col2 in duplicate_cols:
        issues.append({
            "column": f"{col1} & {col2}",
            "issue_type": "Duplicate Columns",
            "description": f"Column **{col2}** has identical data to **{col1}**.",
            "affected_rows": total_rows,
            "severity": "warning"
        })

    # 2. Check for Low Variance / Constant Columns
    for idx, col in enumerate(df.columns):
        col_data = df.iloc[:, idx]
        if col_data.dropna().nunique() <= 1:
            issues.append({
                "column": col,
                "issue_type": "Constant Column",
                "description": f"Column **{col}** contains only 1 unique value (constant).",
                "affected_rows": total_rows,
                "severity": "info"
            })
            
    # 3. Check for Column-level checks: Negative values, blank strings, leading/trailing spaces, mixed types, cardinality
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    object_cols = df.select_dtypes(include=['object', 'str', 'category']).columns
    
    for idx, col in enumerate(df.columns):
        col_data = df.iloc[:, idx]
        # Check Mixed Types
        if col in object_cols:
            types = col_data.dropna().apply(lambda x: type(x).__name__).nunique()
            if types > 1:
                issues.append({
                    "column": col,
                    "issue_type": "Mixed Data Types",
                    "description": f"Column contains mixed Python types.",
                    "affected_rows": "N/A",
                    "severity": "warning"
                })

        # Check Negative values in numeric columns (if they are typically positive)
        # We raise a warning if a numeric column is mostly positive but has a few negatives
        if col in numeric_cols:
            negatives = (col_data < 0).sum()
            if negatives > 0:
                positives = (col_data >= 0).sum()
                if positives > negatives: # It's mostly positive, negatives might be anomalies
                    issues.append({
                        "column": col,
                        "issue_type": "Negative Values",
                        "description": f"Column has {negatives} negative values which may represent errors.",
                        "affected_rows": int(negatives),
                        "severity": "warning"
                    })

        # Check for whitespace issues, empty strings, special characters in string columns
        if col in object_cols:
            s_data = col_data.astype(str).str.strip()
            
            # Blank strings (whitespace only)
            blanks = ((col_data.astype(str) != "") & (s_data == "")).sum()
            if blanks > 0:
                issues.append({
                    "column": col,
                    "issue_type": "Blank Strings",
                    "description": f"Column contains whitespace-only blank values.",
                    "affected_rows": int(blanks),
                    "severity": "info"
                })

            # Leading/Trailing Spaces
            untrimmed = (col_data.astype(str) != s_data).sum()
            if untrimmed > 0:
                issues.append({
                    "column": col,
                    "issue_type": "Untrimmed Spaces",
                    "description": f"Column has leading/trailing spaces that should be trimmed.",
                    "affected_rows": int(untrimmed),
                    "severity": "warning"
                })

            # High Cardinality (e.g. categorical columns with too many unique values)
            if str(col_data.dtype) in ['category', 'object']:
                unique_vals = col_data.nunique()
                if unique_vals > 50 and unique_vals < total_rows * 0.9:
                    issues.append({
                        "column": col,
                        "issue_type": "High Cardinality",
                        "description": f"Column has a high number of unique text values ({unique_vals}), making it difficult to encode.",
                        "affected_rows": "N/A",
                        "severity": "info"
                    })

    # 4. Check for invalid dates in datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        # Check if dates are in the future
        future_dates = (df[col] > pd.Timestamp.now()).sum()
        if future_dates > 0:
            issues.append({
                "column": col,
                "issue_type": "Future Dates",
                "description": f"Column contains {future_dates} dates that are in the future.",
                "affected_rows": int(future_dates),
                "severity": "danger"
            })
            
    return issues
