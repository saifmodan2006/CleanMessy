import pandas as pd
import numpy as np
from typing import List, Union, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler

def apply_math_op(
    df: pd.DataFrame, 
    col1: str, 
    op: str, 
    operand: Union[str, float, int], 
    new_name: str
) -> pd.DataFrame:
    """
    Applies a mathematical operation: df[col1] op operand (operand can be another column name or a constant).
    Operations: 'add', 'subtract', 'multiply', 'divide'
    """
    cleaned_df = df.copy()
    if col1 not in cleaned_df.columns:
        return cleaned_df

    # Resolve operand (either a column value or a float constant)
    if isinstance(operand, str) and operand in cleaned_df.columns:
        val = cleaned_df[operand]
    else:
        try:
            val = float(operand)
        except ValueError:
            raise ValueError(f"Operand '{operand}' must be a column name or a numeric constant.")
            
    if op == 'add':
        cleaned_df[new_name] = cleaned_df[col1] + val
    elif op == 'subtract':
        cleaned_df[new_name] = cleaned_df[col1] - val
    elif op == 'multiply':
        cleaned_df[new_name] = cleaned_df[col1] * val
    elif op == 'divide':
        # Safely handle division by zero
        if isinstance(val, (int, float)) and val == 0:
            cleaned_df[new_name] = np.nan
        else:
            divisor = val.replace(0, np.nan) if isinstance(val, pd.Series) else val
            cleaned_df[new_name] = cleaned_df[col1] / divisor
            
    return cleaned_df

def create_conditional_col(
    df: pd.DataFrame,
    new_col: str,
    cond_col: str,
    op: str,
    value: Any,
    true_val: Any,
    false_val: Any
) -> pd.DataFrame:
    """
    Creates a new column based on condition: df[cond_col] op value -> true_val else false_val.
    ops: '==', '!=', '>', '<', '>=', '<=', 'contains'
    """
    cleaned_df = df.copy()
    if cond_col not in cleaned_df.columns:
        return cleaned_df

    series = cleaned_df[cond_col]
    
    # Cast value if numeric
    try:
        if pd.api.types.is_numeric_dtype(series):
            value = float(value)
    except ValueError:
        pass
        
    if op == '==':
        condition = (series == value)
    elif op == '!=':
        condition = (series != value)
    elif op == '>':
        condition = (series > value)
    elif op == '<':
        condition = (series < value)
    elif op == '>=':
        condition = (series >= value)
    elif op == '<=':
        condition = (series <= value)
    elif op == 'contains':
        condition = series.astype(str).str.contains(str(value), na=False)
    else:
        raise ValueError(f"Unsupported conditional operator: {op}")

    cleaned_df[new_col] = np.where(condition, true_val, false_val)
    return cleaned_df

def extract_date_parts(df: pd.DataFrame, column: str, parts: List[str]) -> pd.DataFrame:
    """
    Extracts parts from a datetime column.
    Parts can include: 'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'Hour'
    """
    cleaned_df = df.copy()
    if column not in cleaned_df.columns:
        return cleaned_df

    # Ensure it is datetime type
    s = pd.to_datetime(cleaned_df[column], errors='coerce')
    
    for part in parts:
        col_name = f"{column}_{part.lower()}"
        if part == 'Year':
            cleaned_df[col_name] = s.dt.year
        elif part == 'Month':
            cleaned_df[col_name] = s.dt.month
        elif part == 'Day':
            cleaned_df[col_name] = s.dt.day
        elif part == 'DayOfWeek':
            cleaned_df[col_name] = s.dt.day_name()
        elif part == 'Quarter':
            cleaned_df[col_name] = s.dt.quarter
        elif part == 'Hour':
            cleaned_df[col_name] = s.dt.hour
            
    return cleaned_df

def calculate_age(df: pd.DataFrame, birth_date_col: str, target_col: str) -> pd.DataFrame:
    """Calculates age based on a birth date column and today's date."""
    cleaned_df = df.copy()
    if birth_date_col in cleaned_df.columns:
        birth_dates = pd.to_datetime(cleaned_df[birth_date_col], errors='coerce')
        today = pd.Timestamp.now()
        # Calculate age in years
        cleaned_df[target_col] = birth_dates.apply(lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day)) if pd.notna(dob) else np.nan)
        # Convert to Int64 (nullable int)
        cleaned_df[target_col] = cleaned_df[target_col].astype('Int64')
    return cleaned_df

def calculate_date_difference(
    df: pd.DataFrame, 
    col1: str, 
    col2: str, 
    new_col: str,
    unit: str = 'days'
) -> pd.DataFrame:
    """Calculates the difference between two date columns (col2 - col1) in specified units."""
    cleaned_df = df.copy()
    if col1 in cleaned_df.columns and col2 in cleaned_df.columns:
        d1 = pd.to_datetime(cleaned_df[col1], errors='coerce')
        d2 = pd.to_datetime(cleaned_df[col2], errors='coerce')
        delta = d2 - d1
        
        if unit == 'days':
            cleaned_df[new_col] = delta.dt.days
        elif unit == 'weeks':
            cleaned_df[new_col] = delta.dt.days / 7.0
        elif unit == 'years':
            cleaned_df[new_col] = delta.dt.days / 365.25
        elif unit == 'hours':
            cleaned_df[new_col] = delta.dt.total_seconds() / 3600.0
            
    return cleaned_df

def apply_log_transform(df: pd.DataFrame, column: str, new_name: str) -> pd.DataFrame:
    """Applies log(x + 1) transformation to a numeric column."""
    cleaned_df = df.copy()
    if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
        # Shift values if they are negative so that minimum value is 0
        min_val = cleaned_df[column].min()
        shift = 0
        if min_val < 0:
            shift = abs(min_val)
        cleaned_df[new_name] = np.log1p(cleaned_df[column] + shift)
    return cleaned_df

def apply_binning(
    df: pd.DataFrame, 
    column: str, 
    num_bins: int, 
    labels: Optional[List[str]], 
    new_name: str,
    strategy: str = 'uniform'
) -> pd.DataFrame:
    """
    Bins a numeric column into categories.
    strategy: 'uniform' (equal width), 'quantile' (equal frequency)
    """
    cleaned_df = df.copy()
    if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
        series = cleaned_df[column].dropna()
        if len(series) > 0:
            if strategy == 'uniform':
                bins = pd.cut(cleaned_df[column], bins=num_bins, labels=labels)
            elif strategy == 'quantile':
                bins = pd.qcut(cleaned_df[column], q=num_bins, labels=labels, duplicates='drop')
            cleaned_df[new_name] = bins
    return cleaned_df

def scale_features(
    df: pd.DataFrame, 
    columns: List[str], 
    scaler_type: str
) -> pd.DataFrame:
    """
    Scales selected numeric columns using scikit-learn preprocessing.
    scaler_type: 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer', 'MaxAbsScaler'
    """
    cleaned_df = df.copy()
    valid_cols = [c for c in columns if c in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[c])]
    
    if len(valid_cols) == 0:
        return cleaned_df
        
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_type == 'RobustScaler':
        scaler = RobustScaler()
    elif scaler_type == 'Normalizer':
        scaler = Normalizer()
    elif scaler_type == 'MaxAbsScaler':
        scaler = MaxAbsScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
        
    # Scale values - fill NaNs with column median before scaling to prevent NaN outputs
    X = cleaned_df[valid_cols].copy()
    for col in valid_cols:
        X[col] = X[col].fillna(X[col].median())
        
    scaled_values = scaler.fit_transform(X)
    cleaned_df[valid_cols] = scaled_values
    
    return cleaned_df
