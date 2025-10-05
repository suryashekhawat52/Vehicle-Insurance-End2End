import os
import sys
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import read_yaml
from src.constants import SCHEMA_FILE_PATH
from src.data_access.vehicle_data import Proj1Data

@dataclass
class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):

        try:
            self.data_ingestion_config = DataIngestionConfig()

        except Exception as e:
            raise CustomException(e,sys)
        
    def export_data_into_feature_store(self)->DataFrame:
        """
        loading data from mongoDB into feature store
        """
        try:
            logging.info("Exporting data from mongoDB")
            my_data = Proj1Data()
            dataframe = my_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"shape of dataframe:{dataframe.shape}")

            self._schema_config = read_yaml(file_path=SCHEMA_FILE_PATH)
            drop_columns = self._schema_config['drop_columns']
            logging.info(f"dropped columns from dataframe {drop_columns}")

            #dropping the id column
            logging.info(dataframe)

            final_dataframe = dataframe.drop(drop_columns, axis = 1)

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Save exported data into feature store file path: {feature_store_file_path}")
            final_dataframe.to_csv(feature_store_file_path, index =False, header=True)
            return final_dataframe
        except Exception as e:
            raise CustomException(e,sys)


    def split_data_as_train_test(self, dataframe:DataFrame) -> None:

        logging.info("Entered split data as train test of data ingestion class")

        try:
            train_set, test_set = train_test_split(dataframe, test_size = self.data_ingestion_config.train_test_split_ratio)
            logging.info("performed train test split on the dataframe")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            print(train_set.columns)

            logging.info(f"Exporting train test file path")

            train_set.to_csv(self.data_ingestion_config.training_file_path, index = False, header = True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index = False, header = True)

            logging.info(f"Exported train test file path")

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        logging.info("Entered initiate data ingestion method of data ingestion class")

        try:
            dataframe = self.export_data_into_feature_store()    
            logging.info("Got the data from mongoDB")
            self.split_data_as_train_test(dataframe)

            logging.info("Performed train test split ")

            data_ingestion_artifact= DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,test_file_path=
                                                           self.data_ingestion_config.testing_file_path)
            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e,sys)
        
        

    