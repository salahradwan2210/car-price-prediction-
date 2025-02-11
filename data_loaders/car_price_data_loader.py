import pandas as pd
import numpy as np
from pandas import DataFrame
import os
from mage_ai.settings.repo import get_repo_path

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def load_data(*args, **kwargs) -> DataFrame:
    """
    Load car price data with optimizations for performance
    """
    try:
        # Get the repo path and construct the full file path
        repo_path = get_repo_path()
        filepath = os.path.join(repo_path, 'raw_data', 'Car_Price.csv')
        
        print(f"Attempting to load data from: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found at {filepath}")
        
        # Read the data
        df = pd.read_csv(filepath)
        print("Initial shape:", df.shape)
        
        # Display column information
        print("\nColumns in the dataset:")
        for col in df.columns:
            print(f"- {col}: {df[col].dtype}")
        
        # Handle missing values
        print("\nHandling missing values...")
        missing_before = df.isnull().sum()
        print("Missing values before cleaning:")
        print(missing_before[missing_before > 0])
        
        # Drop columns with too many missing values (>50%)
        threshold = len(df) * 0.5
        df = df.dropna(axis=1, thresh=threshold)
        
        # Fill missing values
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill numeric columns with median
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                print(f"Filling missing values in {col} with median")
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                print(f"Filling missing values in {col} with mode")
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Verify price column exists and is numeric
        if 'price' not in df.columns:
            raise ValueError("Required 'price' column not found in the dataset")
        
        if not pd.api.types.is_numeric_dtype(df['price']):
            print("Converting price column to numeric...")
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df.dropna(subset=['price'])
        
        print("\nFinal shape:", df.shape)
        print("Memory usage:", df.memory_usage().sum() / 1024**2, "MB")
        
        # Basic statistics of the price column
        print("\nPrice column statistics:")
        print(df['price'].describe())
        
        return df
        
    except Exception as e:
        print(f"Error during data loading: {str(e)}")
        raise

@test
def test_output(df) -> None:
    """
    Test the output of the data loader
    """
    assert df is not None, 'Data is empty'
    assert isinstance(df, DataFrame), 'Output is not a DataFrame'
    assert len(df) > 0, 'DataFrame is empty'
    assert 'price' in df.columns, "'price' column is missing"
    assert pd.api.types.is_numeric_dtype(df['price']), "'price' column must be numeric"
    assert not df.isnull().any().any(), 'Dataset contains missing values'
    print('All tests passed!')

if __name__ == "__main__":
    print("Starting data loading process...")
    df = load_data()
    test_output(df)
    print("Data loading completed successfully!") 