import pandas as pd
from typing import List, Dict, Optional

def rename_columns(df: pd.DataFrame, rename_dict: Dict[str, str]) -> pd.DataFrame:
    """Renames columns in the DataFrame based on a dictionary mapping."""
    return df.rename(columns=rename_dict)

def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Drops specified columns from the DataFrame."""
    return df.drop(columns=columns, errors='ignore')

def duplicate_column(df: pd.DataFrame, column: str, new_name: str) -> pd.DataFrame:
    """Creates a duplicate copy of a column with a new name."""
    cleaned_df = df.copy()
    if column in cleaned_df.columns:
        cleaned_df[new_name] = cleaned_df[column]
    return cleaned_df

def merge_columns(
    df: pd.DataFrame, 
    columns: List[str], 
    new_name: str, 
    separator: str = " "
) -> pd.DataFrame:
    """Merges multiple columns into a single text column separated by a delimiter."""
    cleaned_df = df.copy()
    valid_cols = [c for c in columns if c in cleaned_df.columns]
    if len(valid_cols) > 0:
        cleaned_df[new_name] = cleaned_df[valid_cols].astype(str).agg(separator.join, axis=1)
    return cleaned_df

def split_column(
    df: pd.DataFrame, 
    column: str, 
    separator: str, 
    new_names: List[str]
) -> pd.DataFrame:
    """Splits a single text column by a delimiter into multiple new columns."""
    cleaned_df = df.copy()
    if column in cleaned_df.columns:
        split_data = cleaned_df[column].astype(str).str.split(separator, expand=True)
        # Select or create naming
        for i in range(len(new_names)):
            if i < split_data.shape[1]:
                cleaned_df[new_names[i]] = split_data[i]
            else:
                cleaned_df[new_names[i]] = None
    return cleaned_df

def reorder_columns(df: pd.DataFrame, ordered_cols: List[str]) -> pd.DataFrame:
    """Reorders the columns of the DataFrame based on an ordered list."""
    cleaned_df = df.copy()
    # Check if ordered_cols matches the columns of df (to prevent losing columns)
    existing_cols = [c for c in ordered_cols if c in cleaned_df.columns]
    remaining_cols = [c for c in cleaned_df.columns if c not in existing_cols]
    return cleaned_df[existing_cols + remaining_cols]
