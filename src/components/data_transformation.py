import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_path:str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            numerical_columns = ['year', 'kms_driven']
            categorical_columns = ['name', 'company', 'fuel_type']

            cat_pipe = Pipeline(
                steps=[
                    ("one_hot_encoding", OneHotEncoder(sparse_output=False)),
                    ("scaling", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("numerical_columns", StandardScaler(), numerical_columns),
                    ("categorical_columns", cat_pipe, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys.exc_info())
    
    def initiate_data_transformation(self, raw_data_path):
        '''
        This function is responsible for transforming the data
        '''
        logging.info('Initiating Data Transformation')
        try:
            logging.info('Reading the data from the path parameter')
            data = pd.read_csv(raw_data_path)

            logging.info('Getting the preprocessor object')
            pre_obj = self.get_data_transformation_object()

            logging.info('Separating the input and output df')
            input_feature_df = data.drop(['Price'], axis=1)
            output_feature_df = data['Price']

            logging.info("Applying the preprocessor to the input dataframe")
            input_feature_processed = pre_obj.fit_transform(input_feature_df)
            
            logging.info('Creating a complete array of both input and output')
            data_arr = np.c_[input_feature_processed, np.array(output_feature_df)]

            logging.info('Saving the preprocessor object')
            save_object(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=pre_obj
            )

            logging.info('Returning the new data array')
            return data_arr

        except Exception as e:
            raise CustomException(e, sys.exc_info())