import pandas as pd
import re
from typing import List, Optional

def clean_text_column(
    df: pd.DataFrame, 
    columns: List[str], 
    operation: str, 
    find_str: Optional[str] = None, 
    replace_str: Optional[str] = None
) -> pd.DataFrame:
    """
    Applies text cleaning transformations to specified columns.
    Operations: 'trim', 'remove_extra_spaces', 'remove_punct', 'remove_nums',
                'remove_special', 'upper', 'lower', 'title', 'find_replace', 'replace_missing'
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col not in cleaned_df.columns:
            continue
            
        # Ensure column is treated as string for processing
        s = cleaned_df[col].astype(str)
        # Create mask to preserve original NaN values if we aren't replacing missing text
        nan_mask = cleaned_df[col].isna()
        
        if operation == 'trim':
            s = s.str.strip()
        elif operation == 'remove_extra_spaces':
            s = s.apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
        elif operation == 'remove_punct':
            s = s.apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
        elif operation == 'remove_nums':
            s = s.apply(lambda x: re.sub(r'\d+', '', str(x)))
        elif operation == 'remove_special':
            s = s.apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', '', str(x)))
        elif operation == 'upper':
            s = s.str.upper()
        elif operation == 'lower':
            s = s.str.lower()
        elif operation == 'title':
            s = s.str.title()
        elif operation == 'find_replace':
            if find_str is not None and replace_str is not None:
                s = s.str.replace(find_str, replace_str, regex=False)
        elif operation == 'replace_missing':
            if replace_str is not None:
                # Here we replace actual NaNs or string representations of NaN
                cleaned_df[col] = cleaned_df[col].fillna(replace_str)
                s = cleaned_df[col].astype(str).replace(['nan', 'NaN', 'None', 'None'], replace_str)
                nan_mask = None # Do not restore NaNs
                
        # Re-apply the transformations back to the dataframe
        if nan_mask is not None:
            # Cast back string representation of nan back to pd.NA or np.nan for consistency
            cleaned_df[col] = s.where(~nan_mask, df[col])
        else:
            cleaned_df[col] = s
            
    return cleaned_df
