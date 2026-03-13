import os
import sys
import pandas as pd
import kagglehub
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    """
    Syllabus Focus: Advanced Python (Dataclasses & Config management)
    """
    raw_data_path: str = os.path.join('data', 'raw', "raw_accidents.csv")
    sampled_data_path: str = os.path.join('data', 'sampled', "sampled_accidents.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Step 1: Download from Kaggle
            logging.info("Downloading dataset from Kaggle...")
            path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
            
            # Identify the actual file name in the downloaded path
            raw_file_name = [f for f in os.listdir(path) if f.endswith('.csv')][0]
            downloaded_csv_path = os.path.join(path, raw_file_name)

            # Step 2: Create directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.sampled_data_path), exist_ok=True)

            # Step 3: Reading the raw data (Using Chunking if needed, but here simple read)
            logging.info("Reading the dataset as a DataFrame")
            df = pd.read_csv(downloaded_csv_path)
                
            # Step 4: Saving raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            # Step 5: Resource Optimization (Sampling 10% data for your CPU/GPU)
            logging.info("Creating a 10% sample for development")
            df_sampled = df.sample(frac=0.1, random_state=42)
            df_sampled.to_csv(self.ingestion_config.sampled_data_path, index=False)

            logging.info("Ingestion of the data is completed successfully")

            return (
                self.ingestion_config.sampled_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()