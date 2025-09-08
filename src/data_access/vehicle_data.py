import os 
import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import CustomException 
from src.logger import logging


class Proj1Data:
    """
    Class to export the MongoDB records as pandas dataframe
    """

    def __init__(self) -> None:
        """
        Initialized the mongoDB client connection
        """

        try:
            self.mongo_client = MongoDBClient(database_name = DATABASE_NAME)

        except Exception as e:
            raise CustomException(e,sys)
        
    def export_collection_as_dataframe(self, collection_name:str, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        This function will return the dataframe collected from mongoDB database
        """
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]

            else:
                collection = self.mongo_client[database_name][collection_name]

            logging.info("Fetching data from mongoDB")
            df = pd.DataFrame(list(collection.find()))
            print(f"data collected with length {len(df)}")

            if "id" in df.columns.to_list():
                df = df.drop(columns=["id"], axis = 1)

            df.replace({"na":np.nan}, inplace=True)

            return df 
        except Exception as e:
            raise CustomException(e,sys)

