import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

def encode_categorical(
    df: pd.DataFrame, 
    columns: List[str], 
    method: str, 
    target_column: Optional[str] = None,
    params: Dict[str, Any] = {}
) -> pd.DataFrame:
    """
    Encodes categorical columns using different encoding methods.
    Methods: 'Label Encoding', 'One Hot Encoding', 'Ordinal Encoding',
             'Frequency Encoding', 'Binary Encoding', 'Hash Encoding', 'Target Encoding'
    """
    cleaned_df = df.copy()
    valid_cols = [c for c in columns if c in cleaned_df.columns]
    
    if len(valid_cols) == 0:
        return cleaned_df

    if method == 'Label Encoding':
        for col in valid_cols:
            le = LabelEncoder()
            # Handle NaNs by casting to string first or filling them
            non_nulls = cleaned_df[col].dropna()
            if len(non_nulls) > 0:
                # Store encoded values, keeping NaNs intact
                encoded = le.fit_transform(non_nulls.astype(str))
                cleaned_df.loc[cleaned_df[col].notna(), col] = encoded
                cleaned_df[col] = cleaned_df[col].astype('Int64') # Nullable integer
                
    elif method == 'One Hot Encoding':
        # Drop columns after encoding, standard prefix
        encoder = ce.OneHotEncoder(cols=valid_cols, use_cat_names=True, handle_unknown='value', handle_missing='value')
        cleaned_df = encoder.fit_transform(cleaned_df)
        
    elif method == 'Ordinal Encoding':
        encoder = ce.OrdinalEncoder(cols=valid_cols, handle_unknown='value', handle_missing='value')
        cleaned_df = encoder.fit_transform(cleaned_df)
        
    elif method == 'Frequency Encoding':
        for col in valid_cols:
            freq = cleaned_df[col].value_counts(normalize=True)
            cleaned_df[col] = cleaned_df[col].map(freq)
            
    elif method == 'Binary Encoding':
        encoder = ce.BinaryEncoder(cols=valid_cols, handle_unknown='value', handle_missing='value')
        cleaned_df = encoder.fit_transform(cleaned_df)
        
    elif method == 'Hash Encoding':
        n_components = params.get('n_components', 8)
        encoder = ce.HashingEncoder(cols=valid_cols, n_components=n_components)
        cleaned_df = encoder.fit_transform(cleaned_df)
        
    elif method == 'Target Encoding':
        if not target_column or target_column not in cleaned_df.columns:
            raise ValueError("Target column is required for Target Encoding.")
        
        y = cleaned_df[target_column]
        encoder = ce.TargetEncoder(cols=valid_cols, handle_unknown='value', handle_missing='value')
        cleaned_df = encoder.fit_transform(cleaned_df, y)
        
    return cleaned_df
