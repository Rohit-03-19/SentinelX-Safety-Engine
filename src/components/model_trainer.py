import os
import sys
import pandas as pd
import joblib
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    # Path where the trained model will be saved
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # BYPASSING GRIDSEARCH FOR SPEED
            logging.info("Training single optimized Random Forest for immediate results")
            
            # Using standard robust parameters
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=2,
                n_jobs=-1 # Uses all cores for this SINGLE fit
            )

            model.fit(X_train, y_train)
            
            logging.info("Model training complete. Evaluating...")
            predicted = model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            # SAVING THE MODEL
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(model, self.model_trainer_config.trained_model_file_path)
            
            print(f"✅ Training Complete. Final R2 Score: {r2_square:.4f}")
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)