import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, plot_conf_mat, plot_classification_report
import yaml

# Read the configuration from config/config.yaml
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = config['model']['path']
    mlflow_tracking_uri: str = config['mlflow']['tracking_uri']

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Entered initiate_model_trainer method")

            # Initialize RandomForestClassifier with parameters from config
            model = RandomForestClassifier(**config['model']['params'])
            
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(self.model_trainer_config.mlflow_tracking_uri)
            
            with mlflow.start_run() as run:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Log metrics to MLflow
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                # Log metrics to logs
                logging.info(f"Model Accuracy: {accuracy}")
                logging.info(f"Model Precision: {precision}")
                logging.info(f"Model Recall: {recall}")
                logging.info(f"Model F1 Score: {f1}")


                # Save the trained model locally
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=model
                )
                logging.info(f"Trained model saved at {self.model_trainer_config.trained_model_file_path}")

                # Log the model to MLflow
                mlflow.sklearn.log_model(model, "random_forest_model")
                logging.info(f"Model logged to MLflow with run ID: {run.info.run_id}")

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
