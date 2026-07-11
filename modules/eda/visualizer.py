import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from typing import Dict, Any, Optional, List

def apply_plot_theme(fig: go.Figure, is_dark: bool) -> go.Figure:
    """Applies a clean theme to the Plotly figure matching our UI design system."""
    text_color = "#a1a1aa" if is_dark else "#71717a"
    grid_color = "rgba(255,255,255,0.06)" if is_dark else "rgba(0,0,0,0.06)"
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=text_color, size=11),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=text_color)
        ),
        xaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            tickfont=dict(size=10, color=text_color),
            linecolor=grid_color
        ),
        yaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            tickfont=dict(size=10, color=text_color),
            linecolor=grid_color
        )
    )
    return fig

def get_correlation_matrix_fig(df: pd.DataFrame, is_dark: bool) -> Optional[go.Figure]:
    """Generates an interactive Correlation Heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
        
    corr = numeric_df.corr().round(3)
    
    # Select color scale based on theme
    colorscale = px.colors.diverging.RdBu_r if not is_dark else px.colors.diverging.Picnic
    
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=colorscale,
        labels=dict(color="Correlation"),
        title="Pearson Correlation Heatmap"
    )
    return apply_plot_theme(fig, is_dark)

def get_distribution_fig(df: pd.DataFrame, col: str, is_dark: bool) -> go.Figure:
    """Generates a distribution histogram plot for a numeric column."""
    fig = px.histogram(
        df, 
        x=col, 
        marginal="violin",
        color_discrete_sequence=['#2563eb' if not is_dark else '#818cf8'],
        opacity=0.75,
        title=f"Distribution of {col}"
    )
    return apply_plot_theme(fig, is_dark)

def get_scatter_fig(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    color_col: Optional[str] = None, 
    size_col: Optional[str] = None,
    is_dark: bool = True
) -> go.Figure:
    """Generates an interactive scatter plot."""
    # Ensure sizes are positive numbers for Plotly size mapping
    size_series = None
    if size_col and size_col in df.columns:
        # Scale to avoid errors
        s_min = df[size_col].min()
        size_series = df[size_col]
        if s_min <= 0:
            size_series = size_series - s_min + 1

    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col, 
        size=size_series,
        title=f"{y_col} vs {x_col}",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    return apply_plot_theme(fig, is_dark)

def get_line_fig(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    color_col: Optional[str] = None,
    is_dark: bool = True
) -> go.Figure:
    """Generates an interactive line chart."""
    fig = px.line(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        title=f"{y_col} over {x_col}",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    return apply_plot_theme(fig, is_dark)

def get_bar_fig(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: Optional[str] = None, 
    color_col: Optional[str] = None,
    agg_func: str = "count",
    is_dark: bool = True
) -> go.Figure:
    """Generates an interactive bar chart (with optional aggregation)."""
    if y_col is None or agg_func == "count":
        # Frequency count plot
        val_counts = df[x_col].value_counts().reset_index()
        val_counts.columns = [x_col, 'count']
        fig = px.bar(
            val_counts, 
            x=x_col, 
            y='count',
            color=x_col if color_col == x_col else None,
            title=f"Count of {x_col}",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
    else:
        # Grouped aggregation
        grouped = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
        fig = px.bar(
            grouped, 
            x=x_col, 
            y=y_col,
            title=f"{agg_func.capitalize()} of {y_col} by {x_col}",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
    return apply_plot_theme(fig, is_dark)

def get_pie_fig(df: pd.DataFrame, names_col: str, values_col: Optional[str] = None, is_dark: bool = True) -> go.Figure:
    """Generates an interactive pie chart."""
    if values_col is None:
        counts = df[names_col].value_counts().reset_index()
        counts.columns = [names_col, 'count']
        fig = px.pie(counts, names=names_col, values='count', title=f"Proportions of {names_col}")
    else:
        fig = px.pie(df, names=names_col, values=values_col, title=f"Pie Chart of {values_col} by {names_col}")
    return apply_plot_theme(fig, is_dark)

def get_treemap_fig(df: pd.DataFrame, path_cols: List[str], values_col: Optional[str] = None, is_dark: bool = True) -> go.Figure:
    """Generates a hierarchical Treemap plot."""
    # Ensure all path columns exist
    valid_paths = [c for c in path_cols if c in df.columns]
    if len(valid_paths) == 0:
        raise ValueError("Must specify at least one valid column for path.")
        
    fig = px.treemap(df, path=valid_paths, values=values_col, title="Treemap View")
    return apply_plot_theme(fig, is_dark)

def get_violin_fig(df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str] = None, is_dark: bool = True) -> go.Figure:
    """Generates a violin plot to compare distributions of numeric features across categories."""
    fig = px.violin(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        box=True, 
        points="all", 
        title=f"Distribution of {y_col} by {x_col}"
    )
    return apply_plot_theme(fig, is_dark)

def get_qq_plot_fig(df: pd.DataFrame, col: str, is_dark: bool = True) -> go.Figure:
    """Generates a Q-Q Plot comparing column distribution to normal distribution."""
    series = df[col].dropna()
    if len(series) < 3:
        return go.Figure()
        
    # Get quantiles
    (osm, osr), (slope, intercept, r) = stats.probplot(series, dist="norm")
    
    # Create regression line
    x_line = np.array([osm.min(), osm.max()])
    y_line = slope * x_line + intercept
    
    fig = go.Figure()
    # Add scatter of theoretical quantiles vs sample quantiles
    fig.add_trace(go.Scatter(
        x=osm, 
        y=osr, 
        mode='markers', 
        name='Data points',
        marker=dict(color='#818cf8' if is_dark else '#2563eb', opacity=0.7)
    ))
    # Add reference line
    fig.add_trace(go.Scatter(
        x=x_line, 
        y=y_line, 
        mode='lines', 
        name='Reference Line',
        line=dict(color='#dc2626', width=2)
    ))
    
    fig.update_layout(
        title=f"Normal Q-Q Plot for {col}",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Ordered Values"
    )
    return apply_plot_theme(fig, is_dark)
