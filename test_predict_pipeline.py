import pandas as pd
from src.pipeline.predict_pipeline import PredictionPipeline
from src.logger import logging

if __name__ == "__main__":
    try:
        # Load a sample test dataset
        test_data_path = "DATA/test_data.csv"  # Replace with your test CSV file path
        test_data = pd.read_csv(test_data_path)
        #test_data= test_data_df.drop(columns='Churn', axis=1)
        #ground_truth = test_data_df['Churn']
        
        logging.info("Starting test for prediction pipeline.")
        
        # Initialize prediction pipeline
        prediction_pipeline = PredictionPipeline()
        predictions = prediction_pipeline.predict(test_data)
        
        # Display predictions
        test_data['Predictions'] = predictions
        print(test_data)
        
        logging.info("Prediction pipeline executed successfully.")
    except Exception as e:
        logging.error(f"Error while testing prediction pipeline: {e}")
