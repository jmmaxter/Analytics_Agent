import duckdb
import pandas as pd
import os

def load_and_profile_data(file_path: str) -> dict:
    """
    Reads a CSV/Excel file using DuckDB or Pandas and returns a comprehensive metadata profile.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    if file_path.endswith('.csv'):
        # DuckDB handles CSVs efficiently
        df = duckdb.sql(f"SELECT * FROM read_csv_auto('{file_path}')").df()
    elif file_path.endswith('.xlsx'):
        # Pandas handles Excel files
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

    # Data governance & profiling baseline
    return {
        "row_count": len(df),
        "columns": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values_count": df.isnull().sum().to_dict()
    }