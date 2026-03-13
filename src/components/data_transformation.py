import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
        # Define exactly what we are using
            numerical_columns = ["Hour", "Visibility(mi)", 'Is_Rush_Hour']
            categorical_columns = ['Day',"Weather_Condition"]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Starting Feature Engineering and Column Cleanup")

            for df in [train_df, test_df]:
                # 1. Extract info from the timestamp
                df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
                df['Hour'] = df['Start_Time'].dt.hour
                df['Day'] = df['Start_Time'].dt.day_name()
                
                # Custom feature: Is_Rush_Hour
                df['Is_Rush_Hour'] = df['Hour'].map(lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 19) else 0)
                
                # 2. THE PURGE: Drop raw strings that crash the model
                cols_to_drop = ['ID', 'Description', 'Source', 'Street', 'City', 'Zipcode', 'Country', 'Start_Time']
                df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

            target_column_name = "Severity"
            
            # 1. Separate Features and Target
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # 2. TRANSFORMING
            # We use fit_transform on the WHOLE dataframe to ensure the row count stays 772839
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # 3. THE "BRUTE FORCE" CONCATENATION
            # If input_feature_train_arr came out as a sparse matrix, we convert it to dense
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            logging.info(f"Shape check: Features {input_feature_train_arr.shape}, Target {target_feature_train_df.shape}")

            # Final Join
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)