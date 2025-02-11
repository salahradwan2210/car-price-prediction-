from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from mage_ai.settings.repo import get_repo_path

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def transform(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Prepare features for car price prediction model with improved error handling
    """
    print("Starting feature engineering...")
    print("Input DataFrame shape:", df.shape)
    print("Memory usage before:", df.memory_usage().sum() / 1024**2, "MB")
    
    try:
        # Create a copy of the dataframe to avoid modifying the input
        df_processed = df.copy()
        
        # Create models directory if it doesn't exist
        repo_path = get_repo_path()
        models_dir = os.path.join(repo_path, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Handle categorical variables
        print("\nProcessing categorical variables...")
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        if len(categorical_columns) > 0:
            print(f"Found {len(categorical_columns)} categorical columns:")
            for col in categorical_columns:
                print(f"- {col}: {df_processed[col].nunique()} unique values")
            
            # Save the list of categorical columns
            categorical_columns_path = os.path.join(models_dir, 'categorical_columns.pkl')
            with open(categorical_columns_path, 'wb') as f:
                pickle.dump(list(categorical_columns), f)
            
            # Process each categorical column
            for column in categorical_columns:
                print(f"Encoding {column}...")
                encoder = LabelEncoder()
                # Handle NaN values if any
                df_processed[column] = df_processed[column].fillna('unknown')
                df_processed[column] = encoder.fit_transform(df_processed[column].astype(str))
                
                # Save the encoder
                encoder_path = os.path.join(models_dir, f'label_encoder_{column}.pkl')
                with open(encoder_path, 'wb') as f:
                    pickle.dump(encoder, f)
        
        # Handle numerical features
        print("\nProcessing numerical features...")
        numerical_columns = df_processed.select_dtypes(
            include=['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        ).columns
        
        # Exclude 'price' from scaling if it exists
        if 'price' in numerical_columns:
            numerical_columns = numerical_columns[numerical_columns != 'price']
        
        if len(numerical_columns) > 0:
            print(f"Found {len(numerical_columns)} numerical columns:")
            for col in numerical_columns:
                print(f"- {col}")
            
            # Save the list of numerical columns
            numerical_columns_path = os.path.join(models_dir, 'numerical_columns.pkl')
            with open(numerical_columns_path, 'wb') as f:
                pickle.dump(list(numerical_columns), f)
            
            # Handle any remaining NaN values in numerical columns
            for col in numerical_columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
            # Scale numerical features
            scaler = StandardScaler()
            df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
            
            # Save the scaler
            scaler_path = os.path.join(models_dir, 'standard_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        print("\nFeature engineering summary:")
        print(f"- Processed {len(categorical_columns)} categorical features")
        print(f"- Processed {len(numerical_columns)} numerical features")
        print(f"- Total features: {len(df_processed.columns)}")
        print("Memory usage after:", df_processed.memory_usage().sum() / 1024**2, "MB")
        
        # Save engineered features
        output_path = os.path.join(repo_path, 'data_preprocessed/engineered_features.csv')
        df_processed.to_csv(output_path, index=False)
        print(f"\nEngineered features saved to: {output_path}")
        
        return df_processed
        
    except Exception as e:
        print(f"Error during feature engineering: {str(e)}")
        raise

@test
def test_output(df) -> None:
    """
    Test the output of the transformer
    """
    assert df is not None, 'Data is empty'
    assert isinstance(df, DataFrame), 'Output is not a DataFrame'
    assert len(df) > 0, 'DataFrame is empty'
    
    # Check if all features are numeric after transformation
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    assert len(non_numeric_cols) == 0, f'Some columns are not numeric after transformation: {list(non_numeric_cols)}'
    
    # Check for infinite values
    assert not df.isin([float('inf'), float('-inf')]).any().any(), 'Infinite values found in transformed data'
    
    # Check if all necessary files are saved
    repo_path = get_repo_path()
    models_dir = os.path.join(repo_path, 'models')
    assert os.path.exists(os.path.join(models_dir, 'standard_scaler.pkl')), 'Scaler not saved'
    assert os.path.exists(os.path.join(models_dir, 'categorical_columns.pkl')), 'Categorical columns list not saved'
    assert os.path.exists(os.path.join(models_dir, 'numerical_columns.pkl')), 'Numerical columns list not saved'
    
    print('All tests passed!')

if __name__ == "__main__":
    print("Loading processed data...")
    repo_path = get_repo_path()
    input_path = os.path.join(repo_path, 'processed_data.csv')
    df = pd.read_csv(input_path)
    
    print("Starting feature engineering process...")
    df = transform(df)
    test_output(df)
    print("Feature engineering completed successfully!") 