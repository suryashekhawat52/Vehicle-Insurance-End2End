import sys
from src.exception import CustomException
from src.logger import logging 


from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.entity.config_entity import (DataIngestionConfig,
                                       DataValidationConfig)
from src.entity.artifact_entity import (DataIngestionArtifact, 
                                        DataValidationArtifact)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()



    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Exited the start data ingestion of training pipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:

        try:
            data_validation = DataValidation(data_ingestion_artifact = data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config)
            logging.info("Entered the data validation of training pipeline class")
            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info(
                "performed the data validation and exited the training pipeline class"
            )
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        

    def run_pipeline(self,) -> None:
        """This method of trainingpipeline class is responsible for running complete pipeline
        """

        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)

        except Exception as e:
            raise CustomException(e,sys)