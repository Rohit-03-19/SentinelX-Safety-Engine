import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

class TrainPipeline:
    def __init__(self):
        self.ingestion = DataIngestion()
        self.transformation = DataTransformation()
        self.trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            # 1. Ingestion
            train_path, test_path, _ = self.ingestion.initiate_data_ingestion()

            # 2. Transformation
            train_arr, test_arr, _ = self.transformation.initiate_data_transformation(train_path, test_path)

            # 3. Training
            r2_score = self.trainer.initiate_model_trainer(train_arr, test_arr)
            return r2_score

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    score = pipeline.run_pipeline()
    print(f"Pipeline Training Complete. R2 Score: {score}")