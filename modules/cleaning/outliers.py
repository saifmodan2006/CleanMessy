import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Union
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def detect_outliers_mask(
    df: pd.DataFrame, 
    column: str, 
    method: str, 
    params: Dict[str, Any] = {}
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Detects outliers in a numeric column using various methods.
    Returns: (boolean series mask where True means outlier, dictionary of metadata/thresholds)
    """
    mask = pd.Series(False, index=df.index)
    meta = {}
    
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return mask, meta
        
    series = df[column].dropna()
    if len(series) == 0:
        return mask, meta
        
    if method == 'IQR':
        k = params.get('threshold', 1.5)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        meta = {
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4)
        }
        
    elif method == 'Z Score':
        thresh = params.get('threshold', 3.0)
        mean = series.mean()
        std = series.std()
        if std > 0:
            z_scores = (df[column] - mean) / std
            mask = z_scores.abs() > thresh
            meta = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "lower_bound": round(mean - thresh * std, 4),
                "upper_bound": round(mean + thresh * std, 4)
            }
            
    elif method == 'Modified Z Score':
        # Uses median and Median Absolute Deviation (MAD)
        thresh = params.get('threshold', 3.5)
        median = series.median()
        mad = np.median((series - median).abs())
        if mad > 0:
            mod_z = 0.6745 * (df[column] - median) / mad
            mask = mod_z.abs() > thresh
            meta = {
                "median": round(median, 4),
                "mad": round(mad, 4),
                "lower_bound": round(median - (thresh * mad / 0.6745), 4),
                "upper_bound": round(median + (thresh * mad / 0.6745), 4)
            }
            
    elif method == 'Isolation Forest':
        cont = params.get('contamination', 0.05)
        model = IsolationForest(contamination=cont, random_state=42)
        X = df[[column]].fillna(series.median())
        preds = model.fit_predict(X)
        mask = pd.Series(preds == -1, index=df.index)
        meta = {"contamination": cont}
        
    elif method == 'Local Outlier Factor':
        cont = params.get('contamination', 0.05)
        neighbors = params.get('n_neighbors', 20)
        model = LocalOutlierFactor(n_neighbors=min(neighbors, len(series)-1), contamination=cont)
        X = df[[column]].fillna(series.median())
        preds = model.fit_predict(X)
        mask = pd.Series(preds == -1, index=df.index)
        meta = {"contamination": cont, "n_neighbors": neighbors}
        
    elif method == 'DBSCAN':
        eps = params.get('eps', 0.5)
        min_samples = params.get('min_samples', 5)
        
        # Scale data for DBSCAN since it is distance based
        scaler = StandardScaler()
        X = df[[column]].fillna(series.median())
        X_scaled = scaler.fit_transform(X)
        
        db = DBSCAN(eps=eps, min_samples=min_samples)
        db.fit(X_scaled)
        # -1 in DBSCAN denotes noise/outliers
        mask = pd.Series(db.labels_ == -1, index=df.index)
        meta = {"eps": eps, "min_samples": min_samples}

    return mask, meta

def handle_outliers(
    df: pd.DataFrame, 
    column: str, 
    method: str, 
    action: str, 
    params: Dict[str, Any] = {}
) -> pd.DataFrame:
    """
    Detects outliers and handles them using the specified action: 'Remove', 'Replace', 'Cap'.
    """
    cleaned_df = df.copy()
    if column not in cleaned_df.columns:
        return cleaned_df
        
    mask, meta = detect_outliers_mask(cleaned_df, column, method, params)
    
    if not mask.any():
        return cleaned_df
        
    if action == 'Remove':
        # Drop rows where mask is True
        cleaned_df = cleaned_df[~mask]
    elif action == 'Replace':
        # Replace outliers with median of non-outliers
        non_outlier_median = cleaned_df.loc[~mask, column].median()
        cleaned_df.loc[mask, column] = non_outlier_median
    elif action == 'Cap':
        # Winsorize / Cap values at the lower and upper bounds if available (IQR or Z Score)
        if "lower_bound" in meta and "upper_bound" in meta:
            cleaned_df[column] = cleaned_df[column].clip(lower=meta["lower_bound"], upper=meta["upper_bound"])
        else:
            # Fallback to capping at 1st and 99th percentiles if bounds aren't available (e.g. Isolation Forest)
            lower = cleaned_df[column].quantile(0.01)
            upper = cleaned_df[column].quantile(0.99)
            cleaned_df[column] = cleaned_df[column].clip(lower=lower, upper=upper)
            
    return cleaned_df
