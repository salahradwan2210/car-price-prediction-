FROM python:3.10.1-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["python", "car_price_prediction/transformers/car_price_model_deployment.py"] 