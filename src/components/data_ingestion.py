import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join('artifacts', 'raw_data.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Initiating Data Ingestion')
        try:
            data = pd.read_csv('notebooks/data/refine_car.csv', index_col=[0])
            logging.info('Read the data from data folder to df')

            logging.info('Creating the directory')
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            logging.info('Saving the df into a csv file')
            data.to_csv(self.data_ingestion_config.raw_data_path, index=False)

            logging.info('Data Ingestion in completed')

            return(
                self.data_ingestion_config.raw_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys.exc_info())


# running the code
if __name__ == "__main__":
    obj = DataIngestion()
    raw_path = obj.initiate_data_ingestion()

    data_trans = DataTransformation()
    data_array = data_trans.initiate_data_transformation(raw_path)
    # print(data_array.shape)

    model_trainer = ModelTrainer()
    score = model_trainer.initiate_model_trainer(data_array)
    print(f"Score is {score}")