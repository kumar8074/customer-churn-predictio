import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os
import yaml
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str

class DataIngestion:
    def __init__(self):
        with open("config/config.yaml") as f:
            self.config = yaml.safe_load(f)
        self.ingestion_config = DataIngestionConfig(raw_data_path=self.config['raw_data_path'])

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info('Read the dataset as dataframe')
            return df
        except FileNotFoundError:
            logging.error(f"File not found at: {self.ingestion_config.raw_data_path}")
            raise CustomException(f"File not found at: {self.ingestion_config.raw_data_path}")
        except Exception as e:
            raise CustomException(e)
        
        

 

