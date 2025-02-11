# Car Price Prediction Pipeline

MLOps pipeline for predicting car prices using Mage and FastAPI.

## Project Overview

The project implements a complete MLOps pipeline for car price prediction using Mage.ai, a modern data pipeline tool that makes it easy to orchestrate, transform and analyze data.

### Why Mage?
- **Visual Pipeline Builder**: Mage provides a visual interface to build and monitor data pipelines
- **Python-First**: Write transformations in pure Python with full IDE support
- **Built-in Scheduling**: Schedule pipelines to run automatically
- **Data Quality**: Built-in testing and monitoring capabilities
- **Version Control**: Git integration for code versioning
- **Extensible**: Easy to integrate with other tools and services

## Quick Start Guide

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/salahradwan2210/car-price-prediction-.git
cd car-price-prediction-

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
1. Download the dataset from Kaggle: [Used Cars Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
2. Rename the downloaded file to `Car_Price.csv`
3. Place it in the `raw_data` directory

### 3. Project Structure
```
car_price_prediction/
├── raw_data/              # Place Car_Price.csv here
├── data_preprocessed/     # Engineered features will be saved here
├── models/               # Trained models will be saved here
├── transformers/         # Pipeline transformers
├── data_loaders/        # Data loading scripts
├── templates/           # Web interface templates
└── pipelines/          # Mage pipeline configurations
```

### 4. Running the Pipeline

#### Option 1: Using Mage UI (Recommended)
1. Start Mage server:
```bash
mage start
```

2. Access Mage UI:
- Open http://localhost:6789 in your browser
- Navigate to car_price_pipeline
- You'll see 4 blocks:
  1. Data Loading
  2. Feature Engineering
  3. Model Training
  4. Model Deployment
- Click "Run Pipeline" to execute all steps

3. Monitor Progress:
- Watch real-time execution in the UI
- Check logs for each step
- View data quality metrics

#### Option 2: Using Terminal
1. Run individual components:
```bash
# Data Loading
python data_loaders/car_price_data_loader.py

# Feature Engineering
python transformers/car_price_feature_engineering.py

# Model Training
python transformers/car_price_model_trainer.py

# Start API Server
python transformers/car_price_model_deployment.py
```

### 5. Using the Prediction API

1. After pipeline completion, access:
- Web Interface: http://localhost:8001
- API Endpoint: http://localhost:8001/predict

2. Example API request:
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

### 6. Directory Usage

- `raw_data/`: 
  - Place the original `Car_Price.csv` here
  - This data will be used for initial loading

- `data_preprocessed/`:
  - Contains engineered features
  - Generated after feature engineering step
  - File: `engineered_features.csv`

- `models/`:
  - Stores trained model and artifacts
  - Generated after model training
  - Files:
    - `car_price_model.joblib`: Trained model
    - `model_metrics.json`: Performance metrics
    - `feature_importance.csv`: Feature importance analysis

### 7. Development

- Format code:
```bash
black .
```

- Run tests:
```bash
pytest tests/
```

- Build Docker container:
```bash
docker build -t car-price-prediction .
```

### 8. Monitoring and Logs

- Check Mage UI for pipeline status
- View logs in terminal output
- Monitor API health: http://localhost:8001/health

## Features
- Data ingestion and preprocessing
- Feature engineering
- Model training with Random Forest
- Model deployment with FastAPI
- CI/CD pipeline with GitHub Actions
- Containerization with Docker

## Troubleshooting

1. If data loading fails:
   - Verify `Car_Price.csv` is in `raw_data/`
   - Check file permissions
   - Ensure correct file encoding (UTF-8)

2. If model training fails:
   - Check available memory
   - Verify feature engineering completed
   - Check engineered_features.csv exists

3. If API fails to start:
   - Check port 8001 is available
   - Verify model files exist in models/
   - Check virtual environment is activated
