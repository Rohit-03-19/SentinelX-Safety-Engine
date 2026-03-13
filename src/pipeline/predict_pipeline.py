import sys
import pandas as pd
import os
import joblib
from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        """
        Loads the trained model and preprocessor to make predictions on new data.
        """
        try:
            # Defining paths for the artifacts
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info("Loading model and preprocessor")
            model = joblib.load(model_path)
            
            # If you created a preprocessor (scaler/encoder), load it here
            preprocessor = joblib.load(preprocessor_path)
            
            # Step 1: Transform features if scaling/encoding was used
            data_scaled = preprocessor.transform(features)
            
            # Step 2: Make Prediction
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, Hour, Visibility, Weather_Condition):
        self.Hour = Hour
        self.Visibility = Visibility
        self.Weather_Condition = Weather_Condition

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Hour": [self.Hour],
                "Visibility(mi)": [self.Visibility], # Must match training col name
                "Weather_Condition": [self.Weather_Condition],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)