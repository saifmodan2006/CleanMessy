# CleanMessy - Data Cleaning & Preprocessing Tool

A premium Streamlit-based application for transforming messy datasets into clean, analysis-ready data through intelligent automation and granular manual controls.

## Features

### 🏠 **Home**
- Overview of all available data cleaning and preprocessing tools
- Supported file formats (CSV, Excel, JSON, Parquet)
- Quick access to dataset upload

### 📊 **Data Quality & Analytics**
- **Dashboard & Quality Assessment**: Data quality scoring with actionable recommendations
- **Column Profiling**: Detailed statistical analysis and distribution visualization
- **EDA Charts**: Interactive exploratory data analysis visualizations
- **Report Center**: Automated profiling reports (YData, Sweetviz, Custom)

### 🧹 **Data Cleaning Operations**
- **Data Cleaning**: Missing value handling, duplicate removal, text normalization
- **Type Management**: Automatic data type detection and conversion
- **Outlier Treatment**: Multiple outlier detection and handling methods
- **Column Operations**: Rename, merge, split, reorder columns
- **Row Operations**: Delete, sort, filter, and random sampling

### ⚙️ **Advanced Features**
- **Feature Engineering**: Mathematical operations, conditional columns, date extraction, encoding, scaling
- **Data Validation**: Schema validation, consistency checks, format verification
- **Export / Download**: Multiple export formats (CSV, Excel, JSON, Parquet, Pickle)

## Installation

### Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/saifmodan2006/CleanMessy.git
cd CleanMessy

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Usage

1. **Upload Dataset**: Start by uploading your messy dataset (CSV, Excel, JSON, or Parquet)
2. **Profile & Analyze**: Use Dashboard to understand data quality
3. **Clean & Transform**: Apply cleaning operations from the sidebar menu
4. **Engineer Features**: Create new columns and transformations
5. **Export Results**: Download your cleaned dataset in preferred format

## Project Structure

```
CleanMessy/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── messy_dataset.csv              # Sample dataset for testing
├── assets/
│   └── styles.css                 # Custom styling
├── modules/
│   ├── core/                      # Core utilities & state management
│   ├── cleaning/                  # Data cleaning functions
│   ├── operations/                # Column & row operations
│   ├── engineering/               # Feature engineering
│   ├── validation/                # Data validation
│   └── eda/                       # Exploratory data analysis
└── .streamlit/
    └── config.toml                # Streamlit configuration
```

## Supported File Formats

| Format | Read | Write |
|--------|------|-------|
| CSV/TSV | ✅ | ✅ |
| Excel (.xlsx, .xls) | ✅ | ✅ |
| JSON | ✅ | ✅ |
| Parquet | ✅ | ✅ |
| Pickle | ❌ | ✅ |

## Deployment on Streamlit Cloud

1. Push repository to GitHub
2. Visit [Streamlit Cloud](https://share.streamlit.io/)
3. Create new app from your GitHub repository
4. Set main file path to `app.py`
5. Click Deploy

## Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **ML/Algorithms**: Scikit-learn, Scipy, Statsmodels
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Reporting**: YData-Profiling, Sweetviz
- **Data Encoding**: Category-Encoders

## Features Highlights

✨ **Smart Automation**
- One-click smart clean operations
- Automatic data type detection
- Intelligent imputation strategies

🎯 **Granular Control**
- Column-level transformations
- Row-level filtering and sampling
- Custom mathematical and conditional operations

📊 **Comprehensive Analytics**
- Real-time data quality scoring
- Interactive visualizations
- Automated report generation

🔄 **Full History Management**
- Undo/Redo operations
- Dataset state tracking
- Operation audit logs

## Performance Optimization

- Memory usage optimization for large datasets
- Efficient data type downcasting
- Responsive UI with streaming updates

## License

This project is open-source and available under the MIT License.

## Support

For issues, suggestions, or contributions, please visit the [GitHub Repository](https://github.com/saifmodan2006/CleanMessy).

---

**Transform Messy Data into Clean, Analysis-Ready Datasets.** ✨
