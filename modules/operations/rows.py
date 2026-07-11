import pandas as pd
from typing import List, Union, Optional

def delete_rows(df: pd.DataFrame, indices: List[Union[int, str]]) -> pd.DataFrame:
    """Deletes rows with specified indices from the DataFrame."""
    return df.drop(index=indices, errors='ignore')

def sort_rows(
    df: pd.DataFrame, 
    columns: List[str], 
    ascending: Union[bool, List[bool]] = True
) -> pd.DataFrame:
    """Sorts the DataFrame by specified columns."""
    return df.sort_values(by=columns, ascending=ascending)

def filter_rows_expression(df: pd.DataFrame, expression: str) -> pd.DataFrame:
    """
    Filters rows in the DataFrame using a pandas query expression (e.g., 'Age > 30 & Salary < 50000').
    """
    try:
        return df.query(expression)
    except Exception as e:
        raise ValueError(f"Invalid filter expression '{expression}': {str(e)}")

def random_sample(
    df: pd.DataFrame, 
    n: Optional[int] = None, 
    frac: Optional[float] = None, 
    random_state: int = 42
) -> pd.DataFrame:
    """
    Returns a random sample of rows from the DataFrame.
    Specify either n (number of rows) or frac (fraction of rows).
    """
    if n is not None:
        return df.sample(n=min(n, len(df)), random_state=random_state)
    elif frac is not None:
        return df.sample(frac=frac, random_state=random_state)
    return df
