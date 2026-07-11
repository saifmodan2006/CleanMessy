import pandas as pd
import numpy as np
import io
import os
import streamlit as st
from typing import Tuple, Optional, Dict, Any

@st.cache_data(show_spinner=False)
def _load_dataset_from_bytes(file_bytes: bytes, file_name: str) -> Tuple[pd.DataFrame, str]:
    """Parse uploaded file bytes once and reuse the result across reruns."""
    ext = os.path.splitext(file_name)[1].lower()
    df = pd.DataFrame()
    err = ""

    try:
        buffer = io.BytesIO(file_bytes)
        if ext == '.csv':
            df = pd.read_csv(buffer)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(buffer)
        elif ext == '.tsv':
            df = pd.read_csv(buffer, sep='\t')
        elif ext == '.json':
            df = pd.read_json(buffer)
        elif ext == '.parquet':
            df = pd.read_parquet(buffer)
        else:
            err = f"Unsupported file format: {ext}"
    except Exception as e:
        err = f"Failed to parse file: {str(e)}"

    return df, err

def load_dataset(file_buffer: Any, file_name: str) -> Tuple[pd.DataFrame, str]:
    """
    Loads dataset from file buffer based on the file extension.
    Returns: (DataFrame, Error Message if any)
    """
    if hasattr(file_buffer, "getvalue"):
        file_bytes = file_buffer.getvalue()
    elif isinstance(file_buffer, bytes):
        file_bytes = file_buffer
    else:
        file_bytes = file_buffer.read()

    return _load_dataset_from_bytes(file_bytes, file_name)

def export_dataset(df: pd.DataFrame, format_type: str) -> Tuple[bytes, str]:
    """
    Exports a dataframe to the specified format.
    Returns: (Bytes representation of file, MimeType)
    """
    buffer = io.BytesIO()
    mime = "application/octet-stream"
    format_type = format_type.upper()
    
    if format_type == "CSV":
        csv_str = df.to_csv(index=False)
        return csv_str.encode('utf-8'), "text/csv"
    elif format_type == "EXCEL":
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='CleanMessy_Data')
        buffer.seek(0)
        return buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif format_type == "JSON":
        json_str = df.to_json(orient='records', indent=4)
        return json_str.encode('utf-8'), "application/json"
    elif format_type == "PARQUET":
        df.to_parquet(buffer, index=False, engine='pyarrow')
        buffer.seek(0)
        return buffer.getvalue(), "application/octet-stream"
    elif format_type == "PICKLE":
        df.to_pickle(buffer)
        buffer.seek(0)
        return buffer.getvalue(), "application/octet-stream"
    else:
        raise ValueError(f"Unknown export format: {format_type}")

def optimize_memory(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Optimizes memory usage of the DataFrame by downcasting numeric types
    and converting low-cardinality object columns to the 'category' dtype.

    Uses pd.api.types instead of np.issubdtype so that pandas extension
    types (StringDtype, ArrowDtype, etc.) never cause a TypeError.
    """
    start_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    optimized_df = df.copy()

    for col in optimized_df.columns:
        series = optimized_df[col]
        col_type_str = str(series.dtype)

        # --- Integer columns ---
        if pd.api.types.is_integer_dtype(series):
            try:
                col_min = series.min()
                col_max = series.max()
                if pd.isna(col_min) or pd.isna(col_max):
                    continue
                if col_min >= 0:
                    if col_max < 255:
                        optimized_df[col] = series.astype(np.uint8)
                    elif col_max < 65535:
                        optimized_df[col] = series.astype(np.uint16)
                    elif col_max < 4_294_967_295:
                        optimized_df[col] = series.astype(np.uint32)
                    else:
                        optimized_df[col] = series.astype(np.uint64)
                else:
                    if col_min > -128 and col_max < 127:
                        optimized_df[col] = series.astype(np.int8)
                    elif col_min > -32_768 and col_max < 32_767:
                        optimized_df[col] = series.astype(np.int16)
                    elif col_min > -2_147_483_648 and col_max < 2_147_483_647:
                        optimized_df[col] = series.astype(np.int32)
                    else:
                        optimized_df[col] = series.astype(np.int64)
            except Exception:
                pass  # Leave column unchanged on any unexpected error

        # --- Float columns ---
        elif pd.api.types.is_float_dtype(series):
            try:
                col_min = series.min()
                col_max = series.max()
                if pd.isna(col_min) or pd.isna(col_max):
                    continue
                optimized_df[col] = series.astype(np.float32)
            except Exception:
                pass

        # --- Plain object columns → maybe downcast to category ---
        elif col_type_str == 'object':
            try:
                num_unique = series.nunique()
                if 0 < num_unique < 100 and num_unique < (len(df) * 0.3):
                    optimized_df[col] = series.astype('category')
            except Exception:
                pass

        # All other extension dtypes (StringDtype, BooleanDtype, ArrowDtype…)
        # are left unchanged — no np.issubdtype ever called.

    end_mem = optimized_df.memory_usage(deep=True).sum() / (1024 ** 2)
    saved_mem = start_mem - end_mem
    pct_saved = (saved_mem / start_mem) * 100 if start_mem > 0 else 0

    stats = {
        "start_mem_mb": round(start_mem, 2),
        "end_mem_mb": round(end_mem, 2),
        "saved_mem_mb": round(saved_mem, 2),
        "pct_saved": round(pct_saved, 1),
    }

    return optimized_df, stats
