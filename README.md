# Car Price Prediction Pipeline

MLOps pipeline for predicting car prices using Mage and FastAPI.

## Project Overview
![Pipeline Structure](Screenshot%202025-02-09%20230230.png)

The project implements a complete MLOps pipeline for car price prediction using Mage.ai, a modern data pipeline tool that makes it easy to orchestrate, transform and analyze data.

### Why Mage?
- **Visual Pipeline Builder**: Mage provides a visual interface to build and monitor data pipelines
- **Python-First**: Write transformations in pure Python with full IDE support
- **Built-in Scheduling**: Schedule pipelines to run automatically
- **Data Quality**: Built-in testing and monitoring capabilities
- **Version Control**: Git integration for code versioning
- **Extensible**: Easy to integrate with other tools and services

### Pipeline Components
1. **Data Loading** (`car_price_data_loader.py`):
   - Loads raw car data from CSV
   - Handles data validation and cleaning
   - Prepares data for feature engineering

2. **Feature Engineering** (`car_price_feature_engineering.py`):
   - Transforms raw features into model-ready format
   - Handles categorical encoding
   - Performs feature scaling and normalization

3. **Model Training** (`car_price_model_trainer.py`):
   - Trains Random Forest model
   - Performs cross-validation
   - Saves model metrics and artifacts

4. **Model Deployment** (`car_price_model_deployment.py`):
   - Deploys model with FastAPI
   - Provides REST API endpoints
   - Includes web interface for predictions

## Web Interface
![Web Interface](Screenshot%202025-02-09%20055937.png)

## Features
- Data ingestion and preprocessing
- Feature engineering
- Model training with Random Forest
- Model deployment with FastAPI
- CI/CD pipeline with GitHub Actions
- Containerization with Docker

## Setup and Installation
1. Clone the repository
```bash
git clone https://github.com/salahradwan/car_price_prediction.git
cd car_price_prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run locally
```bash
python car_price_prediction/transformers/car_price_model_deployment.py
```

4. Access the API
- Web UI: http://localhost:8001
- API Endpoint: http://localhost:8001/predict

## Project Structure
```
car_price_prediction/
├── models/                  # Trained models
├── transformers/           # Mage transformers
├── pipelines/             # Data pipelines
├── api/                   # FastAPI application
├── tests/                 # Unit tests
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## API Usage
Example request:
```python
import requests

data = {
    "features": {
        "manufacturer": "Toyota",
        "model": "Camry",
        "year": 2020,
        "condition": "excellent",
        "mileage": 60000,
        "fuel_type": "gas"
    }
}

response = requests.post("http://localhost:8001/predict", json=data)
print(response.json())
```

## Development
- Run tests: `pytest tests/`
- Format code: `black .`
- Build Docker image: `docker build -t car-price-prediction .`

## Running with Mage
1. Start Mage server:
```bash
mage start
```

2. Access Mage UI:
- Open http://localhost:6789 in your browser
- Navigate to car_price_pipeline
- Click "Run Pipeline" to execute the entire workflow

3. Monitor Pipeline:
- View real-time execution status
- Check logs and debugging information
- Monitor data quality metrics 
