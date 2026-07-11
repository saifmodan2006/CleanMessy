import pandas as pd
import numpy as np
import io
import datetime
from typing import Dict, Any, List, Optional
from modules.core.quality import calculate_quality_score, get_improvement_suggestions
from modules.eda.profiling import get_dataset_summary_metrics, get_column_profile
from modules.validation.checker import run_data_validation

# Fallback imports
YDATA_AVAILABLE = False
try:
    from ydata_profiling import ProfileReport
    YDATA_AVAILABLE = True
except ImportError:
    pass

SWEETVIZ_AVAILABLE = False
try:
    import sweetviz as sv
    SWEETVIZ_AVAILABLE = True
except ImportError:
    pass

def generate_ydata_report(df: pd.DataFrame) -> Optional[str]:
    """Generates an HTML report string using ydata-profiling if available."""
    if not YDATA_AVAILABLE:
        return None
    try:
        profile = ProfileReport(df, title="CleanMessy YData Profiling Report", minimal=True)
        return profile.to_html()
    except Exception as e:
        return f"Error generating YData-Profiling report: {str(e)}"

def generate_sweetviz_report(df: pd.DataFrame) -> Optional[str]:
    """Generates an HTML report string using Sweetviz if available."""
    if not SWEETVIZ_AVAILABLE:
        return None
    try:
        # Save to temporary file since Sweetviz outputs directly to disk
        report = sv.analyze(df)
        buffer = io.StringIO()
        report.show_html(filepath="temp_sweetviz_report.html", open_browser=False)
        with open("temp_sweetviz_report.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        try:
            import os
            os.remove("temp_sweetviz_report.html")
        except Exception:
            pass
        return html_content
    except Exception as e:
        return f"Error generating Sweetviz report: {str(e)}"

def generate_custom_cleanmessy_report(df: pd.DataFrame) -> str:
    """
    Generates a beautiful, custom, self-contained HTML profiling and quality report
    if external profiling libraries are not available or as a default premium view.
    """
    summary = get_dataset_summary_metrics(df)
    quality = calculate_quality_score(df)
    suggestions = get_improvement_suggestions(df)
    validation_issues = run_data_validation(df)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build Recommendations list items
    rec_html = ""
    if suggestions:
        for sug in suggestions:
            badge_class = f"badge-{sug['severity']}"
            rec_html += f"""
            <div class="card recommendation-item">
                <span class="badge {badge_class}">{sug['category']}</span>
                <p><strong>Observation:</strong> {sug['message']}</p>
                <p class="action-text"><strong>Recommendation:</strong> {sug['action']}</p>
            </div>
            """
    else:
        rec_html = "<div class='no-issues'>🎉 No significant recommendations. Your dataset looks extremely clean!</div>"

    # Build Validation alerts table
    val_html = ""
    if validation_issues:
        val_html += """
        <table class="data-table">
            <thead>
                <tr>
                    <th>Column(s)</th>
                    <th>Issue Type</th>
                    <th>Details</th>
                    <th>Affected Rows</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>
        """
        for issue in validation_issues:
            badge_class = f"badge-{issue['severity']}"
            val_html += f"""
                <tr>
                    <td><code>{issue['column']}</code></td>
                    <td>{issue['issue_type']}</td>
                    <td>{issue['description']}</td>
                    <td>{issue['affected_rows']}</td>
                    <td><span class="badge {badge_class}">{issue['severity'].upper()}</span></td>
                </tr>
            """
        val_html += "</tbody></table>"
    else:
        val_html = "<div class='no-issues'>🎉 Data validation completed with 0 errors/warnings.</div>"

    # Build Column Profiles section
    cols_html = ""
    for col in df.columns:
        prof = get_column_profile(df, col)
        
        # Stats table rows
        stats_rows = f"""
            <tr><td><strong>Data Type</strong></td><td><code>{prof['dtype']}</code></td></tr>
            <tr><td><strong>Total Rows</strong></td><td>{prof['count']}</td></tr>
            <tr><td><strong>Non-null Count</strong></td><td>{prof['non_null']}</td></tr>
            <tr><td><strong>Missing Count</strong></td><td>{prof['missing']} ({prof['missing_pct']}%)</td></tr>
            <tr><td><strong>Unique Values</strong></td><td>{prof['unique']}</td></tr>
        """
        
        if "mean" in prof: # Numeric
            stats_rows += f"""
                <tr><td><strong>Mean</strong></td><td>{prof['mean']}</td></tr>
                <tr><td><strong>Median</strong></td><td>{prof['median']}</td></tr>
                <tr><td><strong>Min / Max</strong></td><td>{prof['min']} / {prof['max']}</td></tr>
                <tr><td><strong>Std Dev / Variance</strong></td><td>{prof['std']} / {prof['var']}</td></tr>
                <tr><td><strong>Skewness / Kurtosis</strong></td><td>{prof['skew']} / {prof['kurtosis']}</td></tr>
            """
            freq_section = ""
        else: # Categorical
            stats_rows += f"<tr><td><strong>Most Frequent (Mode)</strong></td><td>{prof['mode']}</td></tr>"
            
            freq_items = ""
            for item in prof.get('frequent_values', []):
                freq_items += f"""
                    <tr>
                        <td><code>{item['value']}</code></td>
                        <td>{item['count']}</td>
                        <td>
                            <div class="progress-bar-container">
                                <div class="progress-bar-fill" style="width: {item['pct']}%"></div>
                            </div>
                            <span class="pct-label">{item['pct']}%</span>
                        </td>
                    </tr>
                """
            
            freq_section = f"""
            <div class="col-freq">
                <h5>Top Frequent Values</h5>
                <table class="freq-table">
                    <thead>
                        <tr>
                            <th>Value</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {freq_items}
                    </tbody>
                </table>
            </div>
            """

        cols_html += f"""
        <div class="card col-card">
            <h4>📊 Column: {col}</h4>
            <div class="col-grid">
                <div class="col-stats">
                    <h5>Descriptive Statistics</h5>
                    <table class="stats-table">
                        {stats_rows}
                    </table>
                </div>
                {freq_section}
            </div>
        </div>
        """

    # Self-contained HTML report string
    html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CleanMessy Data Profiling Report</title>
    <style>
        :root {{
            --bg: #09090b;
            --bg-subtle: #0c0c0f;
            --card: #0c0c0f;
            --border: #1e1e24;
            --text: #fafafa;
            --text-muted: #71717a;
            --accent: #2563eb;
            --accent-glow: rgba(37,99,235,0.15);
            --green: #22c55e;
            --green-muted: rgba(34,197,94,0.1);
            --red: #ef4444;
            --red-muted: rgba(239,68,68,0.1);
            --amber: #f59e0b;
            --amber-muted: rgba(245,158,11,0.1);
            --radius: 10px;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            padding: 2rem;
        }}
        
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        h1 {{ font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em; color: var(--text); }}
        .tagline {{ color: var(--text-muted); font-size: 0.95rem; margin-top: 0.2rem; }}
        .timestamp {{ color: var(--text-muted); font-size: 0.85rem; font-family: monospace; }}
        
        h2 {{ font-size: 1.4rem; font-weight: 700; margin-bottom: 1rem; border-left: 4px solid var(--accent); padding-left: 0.6rem; margin-top: 2rem; }}
        h3 {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 0.8rem; }}
        h4 {{ font-size: 1.05rem; font-weight: 600; margin-bottom: 0.5rem; color: #fff; }}
        h5 {{ font-size: 0.85rem; font-weight: 500; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.05em; margin-bottom: 0.6rem; }}
        
        .card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.25rem;
            margin-bottom: 1.25rem;
        }}
        
        /* Grid Layouts */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .kpi-card {{
            text-align: center;
        }}
        .kpi-label {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }}
        .kpi-value {{ font-size: 1.6rem; font-weight: 700; color: #fff; margin-top: 0.2rem; }}
        
        .summary-section {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
        }}
        
        /* Badges */
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-danger {{ color: var(--red); background: var(--red-muted); border: 1px solid rgba(239,68,68,0.2); }}
        .badge-warning {{ color: var(--amber); background: var(--amber-muted); border: 1px solid rgba(245,158,11,0.2); }}
        .badge-info {{ color: var(--accent); background: var(--accent-glow); border: 1px solid rgba(37,99,235,0.2); }}
        .badge-success {{ color: var(--green); background: var(--green-muted); border: 1px solid rgba(34,197,94,0.2); }}
        
        /* Recommendation Card styling */
        .recommendation-item {{
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            padding: 1rem;
            border-left: 3px solid var(--border);
        }}
        .recommendation-item p {{ font-size: 0.85rem; }}
        .action-text {{ color: var(--accent); }}
        
        /* Tables */
        .data-table, .stats-table, .freq-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 0.82rem;
            margin-top: 0.5rem;
        }}
        .data-table th {{
            text-align: left;
            padding: 0.5rem 0.8rem;
            color: var(--text-muted);
            font-weight: 500;
            border-bottom: 1px solid var(--border);
        }}
        .data-table td {{
            padding: 0.6rem 0.8rem;
            border-bottom: 1px solid var(--border);
        }}
        
        .stats-table td {{
            padding: 0.4rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.03);
        }}
        .stats-table tr:last-child td {{ border-bottom: none; }}
        .stats-table td:last-child {{ text-align: right; font-family: monospace; font-size: 0.85rem; color: #fff; }}
        
        .freq-table th {{
            text-align: left;
            padding: 0.4rem;
            font-size: 0.72rem;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
        }}
        .freq-table td {{
            padding: 0.45rem 0.4rem;
            border-bottom: 1px solid rgba(255,255,255,0.03);
        }}
        
        .progress-bar-container {{
            background: rgba(255,255,255,0.04);
            height: 6px;
            width: 70%;
            border-radius: 3px;
            display: inline-block;
            vertical-align: middle;
            margin-right: 8px;
            overflow: hidden;
        }}
        .progress-bar-fill {{
            background: var(--accent);
            height: 100%;
        }}
        .pct-label {{ font-size: 0.75rem; color: var(--text-muted); display: inline-block; vertical-align: middle; }}
        
        .no-issues {{
            padding: 1.5rem;
            background: rgba(255,255,255,0.01);
            border: 1px dashed var(--border);
            border-radius: var(--radius);
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-muted);
        }}
        
        /* Column details cards */
        .col-card {{
            margin-bottom: 1.5rem;
        }}
        .col-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-top: 1rem;
        }}
        @media (max-width: 768px) {{
            .summary-section, .col-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>CleanMessy Data Profile Report</h1>
                <div class="tagline">Transform Messy Data into Clean, Analysis-Ready Datasets.</div>
            </div>
            <div class="timestamp">Generated on: {timestamp}</div>
        </header>
        
        <!-- KPI Row -->
        <div class="kpi-grid">
            <div class="card kpi-card">
                <div class="kpi-label">Quality Score</div>
                <div class="kpi-value" style="color: var(--accent);">{quality['score']}%</div>
            </div>
            <div class="card kpi-card">
                <div class="kpi-label">Rows</div>
                <div class="kpi-value">{summary['rows']}</div>
            </div>
            <div class="card kpi-card">
                <div class="kpi-label">Columns</div>
                <div class="kpi-value">{summary['cols']}</div>
            </div>
            <div class="card kpi-card">
                <div class="kpi-label">Missing Cells</div>
                <div class="kpi-value">{summary['missing_cells']} ({summary['missing_pct']}%)</div>
            </div>
            <div class="card kpi-card">
                <div class="kpi-label">Duplicates</div>
                <div class="kpi-value">{summary['duplicates']}</div>
            </div>
            <div class="card kpi-card">
                <div class="kpi-label">Memory Usage</div>
                <div class="kpi-value">{summary['memory_mb']} MB</div>
            </div>
        </div>
        
        <!-- Summary split -->
        <div class="summary-section">
            <div>
                <h2>⚠️ Validation Alerts</h2>
                {val_html}
            </div>
            <div>
                <h2>💡 Improvements List</h2>
                {rec_html}
            </div>
        </div>
        
        <!-- Columns details -->
        <h2>📋 Individual Column Profiles</h2>
        {cols_html}
    </div>
</body>
</html>
"""
    return html_report
