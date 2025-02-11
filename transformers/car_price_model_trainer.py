from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import json
from mage_ai.settings.repo import get_repo_path

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def transform(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Train and evaluate the car price prediction model
    """
    try:
        print("Starting model training...")
        print("Input DataFrame shape:", df.shape)
        
        # Ensure models directory exists
        repo_path = get_repo_path()
        models_dir = os.path.join(repo_path, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Prepare features and target
        if 'price' not in df.columns:
            raise ValueError("'price' column not found in the dataset")
        
        print("\nPreparing features and target...")
        X = df.drop('price', axis=1)
        y = df['price'].values
        
        # Remove outliers from target variable
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        mask = (y >= lower_bound) & (y <= upper_bound)
        X = X[mask]
        y = y[mask]
        
        print(f"Removed {len(df) - len(X)} outliers")
        print(f"Training data shape after outlier removal: {X.shape}")
        
        # Split the data
        print("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train the model
        print("\nTraining Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        print("\nMaking predictions and calculating metrics...")
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_predictions))),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_predictions))),
            'train_r2': float(r2_score(y_train, train_predictions)),
            'test_r2': float(r2_score(y_test, test_predictions)),
            'train_mae': float(mean_absolute_error(y_train, train_predictions)),
            'test_mae': float(mean_absolute_error(y_test, test_predictions))
        }
        
        # Print model performance
        print("\nModel Performance Metrics:")
        print(f"Train RMSE: ${metrics['train_rmse']:,.2f}")
        print(f"Test RMSE: ${metrics['test_rmse']:,.2f}")
        print(f"Train R2 Score: {metrics['train_r2']:.4f}")
        print(f"Test R2 Score: {metrics['test_r2']:.4f}")
        print(f"Train MAE: ${metrics['train_mae']:,.2f}")
        print(f"Test MAE: ${metrics['test_mae']:,.2f}")
        
        # Save metrics
        metrics_path = os.path.join(models_dir, 'model_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Save feature importance
        feature_importance_path = os.path.join(models_dir, 'feature_importance.csv')
        feature_importance.to_csv(feature_importance_path, index=False)
        
        # Save the model
        model_path = os.path.join(models_dir, 'car_price_model.joblib')
        joblib.dump(model, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Add predictions to the input DataFrame
        df_with_predictions = df.copy()
        df_with_predictions['predicted_price'] = model.predict(df_with_predictions.drop('price', axis=1))
        
        return df_with_predictions
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

@test
def test_output(df) -> None:
    """
    Test the model training output
    """
    assert df is not None, 'Output DataFrame is empty'
    assert isinstance(df, DataFrame), 'Output is not a DataFrame'
    assert 'predicted_price' in df.columns, 'Predictions column is missing'
    
    # Check if model files exist
    repo_path = get_repo_path()
    models_dir = os.path.join(repo_path, 'models')
    
    assert os.path.exists(os.path.join(models_dir, 'car_price_model.joblib')), 'Model file not created'
    assert os.path.exists(os.path.join(models_dir, 'model_metrics.json')), 'Metrics file not created'
    assert os.path.exists(os.path.join(models_dir, 'feature_importance.csv')), 'Feature importance file not created'
    
    # Load and check metrics
    with open(os.path.join(models_dir, 'model_metrics.json'), 'r') as f:
        metrics = json.load(f)
    assert metrics['test_r2'] > -1, 'Model performance is extremely poor'
    
    print('All tests passed!')

if __name__ == "__main__":
    print("Loading engineered features...")
    repo_path = get_repo_path()
    input_path = os.path.join(repo_path, 'engineered_features.csv')
    df = pd.read_csv(input_path)
    
    print("Starting model training process...")
    df = transform(df)
    test_output(df)
    print("Model training completed successfully!") 