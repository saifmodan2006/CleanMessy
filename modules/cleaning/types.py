import pandas as pd
from typing import Dict, Any, Optional

def auto_detect_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infers the logical datatype of each column in the dataframe.
    Returns a mapping of column name to logical type: 'Integer', 'Float', 'Boolean', 'Datetime', 'Category', 'Text/Object'
    """
    dtypes = {}
    for col in df.columns:
        # Check if datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dtypes[col] = 'Datetime'
            continue
            
        # Try parsing as datetime if object type
        if df[col].dtype == 'object':
            try:
                # We check a subset of non-null values to speed up and avoid false positives
                subset = df[col].dropna().head(100)
                if len(subset) > 0:
                    parsed = pd.to_datetime(subset, errors='raise')
                    dtypes[col] = 'Datetime'
                    continue
            except (ValueError, TypeError):
                pass

        # Check boolean
        if pd.api.types.is_bool_dtype(df[col]):
            dtypes[col] = 'Boolean'
            continue
        elif df[col].nunique() == 2:
            # Check if it has boolean-like values e.g. [0, 1] or ['Yes', 'No'] or ['T', 'F']
            vals = set(df[col].dropna().unique())
            if vals.issubset({0, 1, '0', '1', 'True', 'False', 'true', 'false', 'T', 'F', 'Yes', 'No', 'yes', 'no'}):
                dtypes[col] = 'Boolean'
                continue

        # Check numeric
        if pd.api.types.is_integer_dtype(df[col]):
            dtypes[col] = 'Integer'
        elif pd.api.types.is_float_dtype(df[col]):
            dtypes[col] = 'Float'
        # Check categorical (low cardinality object)
        elif isinstance(df[col].dtype, pd.CategoricalDtype):
            dtypes[col] = 'Category'
        elif df[col].dtype == 'object':
            unique_count = df[col].nunique()
            if 0 < unique_count < 20 or (unique_count < len(df) * 0.1):
                dtypes[col] = 'Category'
            else:
                dtypes[col] = 'Text/Object'
        else:
            dtypes[col] = str(df[col].dtype)
            
    return dtypes

def convert_type(
    df: pd.DataFrame, 
    column: str, 
    target_type: str, 
    date_format: Optional[str] = None
) -> pd.DataFrame:
    """
    Converts a column to a specified target type.
    target_type options: 'Integer', 'Float', 'Boolean', 'Category', 'Object', 'Datetime'
    """
    cleaned_df = df.copy()
    if column not in cleaned_df.columns:
        return cleaned_df

    try:
        if target_type == 'Integer':
            # Handle float NaNs by first using pd.Int64Dtype (nullable integer)
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce').round().astype('Int64')
        elif target_type == 'Float':
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce').astype(float)
        elif target_type == 'Boolean':
            # Map common strings to boolean
            mapped = cleaned_df[column].astype(str).str.strip().str.lower()
            true_vals = {'true', '1', '1.0', 'yes', 't', 'y'}
            false_vals = {'false', '0', '0.0', 'no', 'f', 'n'}
            
            def parse_bool(val):
                if val in true_vals:
                    return True
                if val in false_vals:
                    return False
                return None
                
            bool_series = mapped.apply(parse_bool)
            cleaned_df[column] = bool_series.astype('boolean') # Nullable boolean
        elif target_type == 'Category':
            cleaned_df[column] = cleaned_df[column].astype('category')
        elif target_type == 'Object':
            cleaned_df[column] = cleaned_df[column].astype(str).replace('nan', None)
        elif target_type == 'Datetime':
            if date_format:
                cleaned_df[column] = pd.to_datetime(cleaned_df[column], format=date_format, errors='coerce')
            else:
                cleaned_df[column] = pd.to_datetime(cleaned_df[column], errors='coerce')
    except Exception as e:
        raise ValueError(f"Could not convert column '{column}' to {target_type}: {str(e)}")
        
    return cleaned_df
