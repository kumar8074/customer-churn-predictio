from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

if __name__ == "__main__":
    try:
        # Test Data Ingestion
        logging.info("Testing Data Ingestion Component")
        data_ingestion = DataIngestion()
        data = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion Completed: Loaded {data.shape[0]} rows and {data.shape[1]} columns")

        # Test Data Transformation
        logging.info("Testing Data Transformation Component")
        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(data)
        logging.info(f"Data Transformation Completed: Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Test Model Trainer
        logging.info("Testing Model Trainer Component")
        model_trainer = ModelTrainer()
        accuracy = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
        logging.info(f"Model Training Completed with Accuracy: {accuracy:.2f}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
