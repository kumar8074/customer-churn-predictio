import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import os
import joblib

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    test_data_path = os.path.join('data', 'test_data.csv')  # Path to save test data

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, df):
        try:
            logging.info('Data Transformation initiated')
            
            # Separate features and target
            X = df.drop(columns='Churn', axis=1)
            y = df['Churn']
            
            # Scale the features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            logging.info('Data Transformation Completed')

            # Ensure the artifacts and data directories exist
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.test_data_path), exist_ok=True)

            # Save the preprocessor object
            joblib.dump(scaler, self.data_transformation_config.preprocessor_obj_file_path)
            logging.info(f"Preprocessor object saved at {self.data_transformation_config.preprocessor_obj_file_path}")

            # Save test data as a CSV file
            test_data = pd.DataFrame(X_test, columns=X.columns)
            #test_data['Churn'] = y_test.values  # No need to Add the target column back
            test_data.to_csv(self.data_transformation_config.test_data_path, index=False)
            logging.info(f"Test data saved at {self.data_transformation_config.test_data_path}")

            return (
                X_train,
                X_test,
                y_train,
                y_test
            )
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e)
