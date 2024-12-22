from src.pipeline.train_pipeline import train_pipeline
from src.logger import logging

if __name__ == "__main__":
    try:
        logging.info("Starting test for training pipeline.")
        train_pipeline()
        logging.info("Training pipeline executed successfully.")
    except Exception as e:
        logging.error(f"Error while testing training pipeline: {e}")
