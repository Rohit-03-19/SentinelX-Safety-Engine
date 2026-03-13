import sys
import os
# Ensures the 'src' directory is in the path for modular imports
sys.path.append(os.getcwd()) 

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

if __name__ == "__main__":
    try:
        # 1. Data Ingestion
        logging.info("Step 1: Data Ingestion")
        ingestion = DataIngestion()
        ingestion_output = ingestion.initiate_data_ingestion()
        
        # --- SAFE UNPACKING LOGIC ---
        # If it's a string, it means ingestion returned only ONE path
        if isinstance(ingestion_output, str):
            train_data_path = ingestion_output
            test_data_path = ingestion_output # Fallback
            logging.warning("Ingestion returned a single string. Check if test data is missing.")
        else:
            # If it's a tuple/list, we grab the paths correctly
            train_data_path = ingestion_output[0]
            test_data_path = ingestion_output[1]
        # ----------------------------

        logging.info(f"Paths verified: Train -> {train_data_path}, Test -> {test_data_path}")

        # 2. Data Transformation
        logging.info("Step 2: Data Transformation")
        transformation = DataTransformation()
        
        # We pass the paths we just verified
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_data_path, 
            test_data_path
        )

        # 3. Model Training
        logging.info("Step 3: Model Training")
        trainer = ModelTrainer()
        score = trainer.initiate_model_trainer(train_arr, test_arr)
        
        print("\n" + "⭐" * 50)
        print(f"🚀 PROJECT SENTINEL-X IS LIVE!")
        print(f"📊 Final R2 Score: {score:.4f}")
        print("⭐" * 50)

    except Exception as e:
        logging.error("Critical failure in the main execution pipeline")
        raise CustomException(e, sys)