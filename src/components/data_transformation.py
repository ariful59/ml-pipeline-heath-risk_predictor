from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import Error
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function creates the data transformation pipeline
        '''
        try:
            # Columns (target and Patient_ID are excluded before transform)
            numerical_columns = ['Age', 'Symptom_Count']
            categorical_columns = [
                'Gender',
                'Lifestyle',
                'Medical_History',
                'Medications',
                'Lab_Reports',
                'Diagnosis',
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise Error(e)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Risk_Level"
            # Drop free-text columns after feature engineering
            drop_columns = [target_column_name, "Patient_ID", "Doctor_Notes", "Symptoms"]

            # Feature Engineering: Create Symptom_Count BEFORE dropping Symptoms
            logging.info("Applying Feature Engineering: Symptom_Count")
            for df in [train_df, test_df]:
                if 'Symptoms' in df.columns:
                    df['Symptom_Count'] = df['Symptoms'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

            input_feature_train_df = train_df.drop(columns=drop_columns)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Target Encoding (Ordinal: Low=0, Medium=1, High=2 for simplicity/mapping)
            # Or use LabelEncoder. Let's do simple mapping to be safe
            risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
            target_train_arr = target_feature_train_df.map(risk_map).values
            target_test_arr = target_feature_test_df.map(risk_map).values

            train_arr = np.c_[
                input_feature_train_arr, target_train_arr
            ]
            test_arr = np.c_[
                input_feature_test_arr, target_test_arr
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise Error(e)
