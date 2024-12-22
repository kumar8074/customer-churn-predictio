import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
from src.utils import load_object
from src.logger import logging

class PredictionPipeline:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def predict(self, data):
        try:
            # Load preprocessor and model
            logging.info("Loading preprocessor and model")
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            # Preprocess input data
            logging.info("Preprocessing input data")
            processed_data = preprocessor.transform(data)

            # Make predictions
            logging.info("Making predictions")
            predictions = model.predict(processed_data)
            return predictions

        except Exception as e:
            logging.error(f"Error in prediction pipeline: {e}")
            raise e
