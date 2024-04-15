import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            print("After Loading")

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e, sys.exc_info())

class CustomData:
    def __init__(self, name, company, year, kms_driven, fuel_type):
        self.name = name
        self.company = company
        self.year = year
        self.kms_driven = kms_driven
        self.fuel_type = fuel_type

    def get_data_as_dataframe(self):
        try:
            custom_data_dict = {
                'name': [self.name],
                'company': [self.company],
                'year': [self.year],
                'kms_driven': [self.kms_driven],
                'fuel_type': [self.fuel_type]
            }
            return pd.DataFrame(custom_data_dict)
        except CustomException as e:
            raise CustomException(e, sys.exc_info())