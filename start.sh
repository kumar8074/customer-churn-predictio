#!/bin/sh

# Start MLflow UI in the background
mlflow ui --host 0.0.0.0 --port 5000 &

# Start Flask app
flask run --host=0.0.0.0 --port=5001