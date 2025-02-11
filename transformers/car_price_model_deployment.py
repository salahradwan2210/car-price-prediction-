from pandas import DataFrame
import pandas as pd
import numpy as np
import joblib
import os
from mage_ai.settings.repo import get_repo_path
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import threading
import webbrowser
import time
from sklearn import preprocessing

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

# Initialize FastAPI app
app = FastAPI(
    title="Car Price Prediction API",
    description="API for predicting car prices",
    version="1.0.0"
)

# Set up templates directory
repo_path = get_repo_path()
templates_dir = os.path.join(repo_path, 'templates')
templates = Jinja2Templates(directory=templates_dir)

# Global variables for model and data
model = None
model_features = None

def load_model_and_preprocessors():
    """Load the trained model and preprocessors"""
    global model, model_features
    try:
        # Get the model path using the workspace root
        repo_path = get_repo_path()
        model_path = os.path.join(repo_path, 'car_price_prediction', 'models', 'car_price_model.joblib')
        print(f"Looking for model at: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Model not found at primary path, trying alternative path...")
            # Try alternative path
            model_path = os.path.join(repo_path, 'models', 'car_price_model.joblib')
            print(f"Looking for model at: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"Model not found at alternative path, trying current directory...")
                # Try one more path
                model_path = os.path.join(os.getcwd(), 'car_price_prediction', 'models', 'car_price_model.joblib')
                print(f"Looking for model at: {model_path}")
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model not found in any of the expected locations")
        
        print(f"Found model at: {model_path}")
        print("Loading model file...")
        # Load model and preprocessors
        loaded_data = joblib.load(model_path)
        if isinstance(loaded_data, dict):
            model = loaded_data['model']
            if 'feature_names' in loaded_data:
                model_features = loaded_data['feature_names']
            print("Model and preprocessors loaded successfully")
        else:
            model = loaded_data
            print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        print("Full error:")
        print(traceback.format_exc())
        return False

def find_free_port(start_port=8000, max_port=8100):
    """Find a free port to use"""
    import socket
    for port in range(start_port, max_port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('0.0.0.0', port))
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError('Could not find a free port')

def start_fastapi():
    """Start the FastAPI server"""
    try:
        # Find an available port
        port = find_free_port()
        print(f"\nStarting server on port {port}...")
        
        # Kill any existing process on the port (Windows-specific)
        import subprocess
        subprocess.run(['taskkill', '/F', '/PID', str(os.getpid())], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        
        # Run the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            workers=1
        )
    except Exception as e:
        print(f"Error starting FastAPI server: {str(e)}")
        raise

def open_browser():
    """Open the browser after a short delay"""
    time.sleep(2)  # Wait for server to start
    port = find_free_port(start_port=8000, max_port=8100)
    webbrowser.open(f'http://localhost:{port}')

class CarFeatures(BaseModel):
    features: dict

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(car_features: CarFeatures):
    """Make predictions using the loaded model"""
    try:
        print("\n=== Debug Information ===")
        print("Input features:", car_features.features)
        
        # Create input DataFrame with all required features
        input_data = pd.DataFrame([{
            'id': 0,  # Placeholder ID
            'url': 'unknown',
            'region': 'unknown',
            'region_url': 'unknown',
            'year': float(car_features.features.get('year', 0)),
            'manufacturer': str(car_features.features.get('manufacturer', '')).lower(),
            'model': str(car_features.features.get('model', '')).lower(),
            'condition': str(car_features.features.get('condition', 'good')).lower(),
            'cylinders': str(car_features.features.get('cylinders', '4')),
            'fuel': str(car_features.features.get('fuel_type', 'gas')).lower(),
            'odometer': float(car_features.features.get('mileage', 0)),
            'title_status': 'clean',
            'transmission': str(car_features.features.get('transmission', 'automatic')).lower(),
            'VIN': 'unknown',
            'drive': str(car_features.features.get('drive', 'fwd')).lower(),
            'type': str(car_features.features.get('type', 'sedan')).lower(),
            'paint_color': str(car_features.features.get('paint_color', 'unknown')).lower(),
            'image_url': 'unknown',
            'description': f"{car_features.features.get('year', '')} {car_features.features.get('manufacturer', '')} {car_features.features.get('model', '')}",
            'state': 'unknown',
            'lat': 0.0,
            'long': 0.0,
            'posting_date': '2024-02-09'  # Current date as default
        }])
        
        print("\nOriginal DataFrame:")
        print(input_data[['year', 'manufacturer', 'model', 'odometer', 'fuel', 'transmission']].to_string())
        
        # Convert categorical columns to numeric using label encoding
        categorical_columns = [
            'url', 'region', 'region_url', 'manufacturer', 'model', 'condition',
            'cylinders', 'fuel', 'title_status', 'transmission', 'VIN', 'drive',
            'type', 'paint_color', 'image_url', 'description', 'state',
            'posting_date'
        ]
        
        # Create label encoders for each categorical column
        label_encoders = {}
        for col in categorical_columns:
            if col in input_data.columns:
                label_encoders[col] = preprocessing.LabelEncoder()
                input_data[col] = label_encoders[col].fit_transform(input_data[col])
        
        # Ensure numeric fields are float and scale them
        numeric_columns = ['id', 'year', 'odometer', 'lat', 'long']
        for col in numeric_columns:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype(float)
        
        # Scale year to be between 0 and 1 based on reasonable range
        input_data['year'] = (input_data['year'] - 1900) / (2024 - 1900)
        
        # Scale odometer based on reasonable maximum value (300,000 miles)
        input_data['odometer'] = input_data['odometer'] / 300000.0
        
        print("\nProcessed DataFrame:")
        print(input_data.to_string())
        print("\nColumns:", input_data.columns.tolist())
        
        # Make prediction
        if model is None:
            load_model_and_preprocessors()
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Calculate confidence interval (using 2 standard deviations for 95% confidence)
        tree_predictions = np.array([tree.predict(input_data)[0] for tree in model.estimators_])
        prediction_std = np.std(tree_predictions)
        
        lower_bound = max(0, prediction - (2 * prediction_std))
        upper_bound = prediction + (2 * prediction_std)
        
        print("\nPrediction Details:")
        print(f"Predicted Price: ${prediction:,.2f}")
        print(f"Price Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
        print("=== End Debug Information ===\n")
        
        return {
            "predicted_price": prediction,
            "predicted_price_formatted": f"${prediction:,.2f}",
            "price_range": f"${lower_bound:,.2f} - ${upper_bound:,.2f}"
        }
        
    except ValueError as e:
        print("ValueError:", str(e))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {str(e)}. Please ensure all numeric fields contain valid numbers."
        )
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@transformer
def execute_transformer_steps(data, *args, **kwargs):
    """Execute the transformer steps"""
    print("Starting car price prediction service...")
    
    # Load model when transformer starts
    if load_model_and_preprocessors():
        print("Model loaded successfully, starting server...")
        
        # Configure and start the server
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=8001,
            reload=False,
            workers=1,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # Start server in a separate thread
        server_thread = threading.Thread(target=server.run)
        server_thread.daemon = True
        server_thread.start()
        
        # Open browser in a separate thread
        def open_browser():
            time.sleep(3)  # Wait a bit longer for server to start
            print("Opening browser...")
            webbrowser.open('http://127.0.0.1:8001')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Keep the transformer running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down server...")
    else:
        print("Failed to load model. Server not started.")
    
    return data

@test
def test_output(df) -> None:
    """
    Test the deployment output
    """
    try:
        assert df is not None, 'Output DataFrame is None'
        assert isinstance(df, DataFrame), 'Output is not a DataFrame'
        assert len(df) > 0, 'Output DataFrame is empty'
        assert 'predicted_price' in df.columns, 'Predictions column is missing'
        assert 'price_lower_bound' in df.columns, 'Lower bound column is missing'
        assert 'price_upper_bound' in df.columns, 'Upper bound column is missing'
        print('All tests passed successfully!')
    except Exception as e:
        print(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    print("Please run this through Mage using:")
    print("mage run car_price_prediction car_price_pipeline")