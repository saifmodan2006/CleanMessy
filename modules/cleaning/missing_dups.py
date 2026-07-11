import pandas as pd
from typing import List, Optional, Any, Union

def impute_missing(
    df: pd.DataFrame, 
    columns: List[str], 
    strategy: str, 
    custom_value: Optional[Any] = None
) -> pd.DataFrame:
    """
    Imputes missing values in specified columns using a given strategy.
    Strategies: 'mean', 'median', 'mode', 'ffill', 'bfill', 'constant'
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col not in cleaned_df.columns:
            continue
            
        if strategy == 'mean':
            # Mean is only applicable for numeric columns
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                fill_val = cleaned_df[col].mean()
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                fill_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
        elif strategy == 'mode':
            mode_series = cleaned_df[col].mode()
            if not mode_series.empty:
                fill_val = mode_series.iloc[0]
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
        elif strategy == 'ffill':
            cleaned_df[col] = cleaned_df[col].ffill()
        elif strategy == 'bfill':
            cleaned_df[col] = cleaned_df[col].bfill()
        elif strategy == 'constant':
            if custom_value is not None:
                # Cast custom value if numeric
                try:
                    if pd.api.types.is_integer_dtype(cleaned_df[col]):
                        custom_value = int(custom_value)
                    elif pd.api.types.is_float_dtype(cleaned_df[col]):
                        custom_value = float(custom_value)
                except ValueError:
                    pass
                cleaned_df[col] = cleaned_df[col].fillna(custom_value)
                
    return cleaned_df

def drop_missing_rows(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Drops rows containing missing values in the specified columns."""
    return df.dropna(subset=columns)

def remove_duplicates(
    df: pd.DataFrame, 
    subset: Optional[List[str]] = None, 
    keep: Union[str, bool] = 'first'
) -> pd.DataFrame:
    """
    Removes duplicate rows in the dataframe.
    keep: 'first', 'last', False (drops all duplicates)
    """
    # Parse keep parameter
    if keep == 'none' or keep is False:
        keep_val = False
    else:
        keep_val = keep
        
    return df.drop_duplicates(subset=subset, keep=keep_val)

def drop_empty_rows_cols(
    df: pd.DataFrame, 
    drop_rows: bool = True, 
    drop_cols: bool = True
) -> pd.DataFrame:
    """
    Automatically drops rows or columns that are completely empty (all NaN).
    """
    cleaned_df = df.copy()
    if drop_rows:
        cleaned_df = cleaned_df.dropna(how='all')
    if drop_cols:
        cleaned_df = cleaned_df.dropna(how='all', axis=1)
    return cleaned_df
