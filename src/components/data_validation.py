import json 
import yaml 
import sys 
import os 

import pandas as pd
from pandas import DataFrame

from src.logger import logging 
from src.exception import CustomException 
from src.utils.main_utils import read_yaml, write_yaml 
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from src.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise CustomException(e,sys)
        
    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        method name: validate number of columns
        Objective: verify the number of columns used are same in the defined schema and ingested data
        Output: returns the bool value based on validation results
        """
        try:
            status = len(dataframe.columns) == len(self.schema_config['columns'])
            logging.info(f"Is required columns presend {status}")
            logging.info(f"lenth of columns present {len(dataframe.columns)}")
            return status
        except Exception as e:
            raise CustomException(e,sys)
        
    def is_column_exists(self, dataframe: DataFrame) -> bool:
        """
        Validate the presence of columns present in schema with Ingested data columns
        """
        try:
            num_columns_missing = []
            cat_columns_missing = []
            for col in self.schema_config["numerical_columns"]:
                if col not in dataframe.columns:
                    num_columns_missing.append(col)

            if len(num_columns_missing) > 0:
                logging.info(f"Numerical columns missing: {num_columns_missing}")

            for col in self.schema_config["categorical_columns"]:
                if col not in dataframe.columns:
                    cat_columns_missing.append(col)
            
            if len(cat_columns_missing) > 0:
                logging.info(f"categorical columns missing: {cat_columns_missing}")

            return False if len(num_columns_missing) > 0 or len(cat_columns_missing) > 0 else True
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def read_data(file_path:str)-> DataFrame:
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method: Initiating data validation  
        """
        try:
            validation_error_msg = ""
            train_df, test_df = (DataValidation.read_data(file_path = self.data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path = self.data_ingestion_artifact.test_file_path))
            
            status = self.validate_number_of_columns(train_df)
            if not status:
                validation_error_msg+= "Missing columns in train_df"
            else:
                logging.info(f"All columns present in train_df:{status}")

            status = self.validate_number_of_columns(test_df)
            if not status:
                validation_error_msg+= "Missing columns in test_df"
            else:
                logging.info(f"All columns present in test_df: {status}")

            #validation col dtype for train/test df
            status = self.is_column_exists(train_df)
            if not status:
                validation_error_msg+= f"Columns are missing in training df"
            else:
                logging.info(f"All categorical/int columns are present in training df: {status}")

            status = self.is_column_exists(test_df)
            if not status:
                validation_error_msg+= f"Columns are missing in testing df"
            else:
                logging.info(f"All categorical/int columns are present in testing df: {status}")
            
            validation_status = len(validation_error_msg) ==0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            #Ensure the directory for validation report file path exists
            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            validation_report = {
                "validation_status": validation_status,
                "error_message": validation_error_msg.strip()
            }

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file,indent=4)

            logging.info("Data validation artifact created and saved to json file")
            logging.info(f"Data validation artifact:{data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)
