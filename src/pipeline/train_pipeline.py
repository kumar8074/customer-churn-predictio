import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging




def train_pipeline():
    try:
        # Data Ingestion
        logging.info("Starting Data Ingestion")
        data_ingestion = DataIngestion()
        data = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        # Data Transformation
        logging.info("Starting Data Transformation")
        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(data)
        logging.info("Data Transformation Completed")

        # Model Training
        logging.info("Starting Model Training")
        model_trainer = ModelTrainer()
        accuracy = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
        logging.info(f"Model Training Completed with Accuracy: {accuracy}")

    except Exception as e:
        logging.error(f"Error in training pipeline: {e}")
        raise e

if __name__ == "__main__":
    train_pipeline()
