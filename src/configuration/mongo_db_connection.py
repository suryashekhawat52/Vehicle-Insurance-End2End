import os 
import sys
import pymongo
import certifi

from src.exception import CustomException
from src.logger import logging 
from src.constants import DATABASE_NAME, MONGODB_URL_KEY

ca = certifi.where()

class MongoDBClient:

    """
    MongoDBClient is responsible for establishing a connection to the mongoDB database.

    Attributes:
    ----------
    client: MongoClient
        A shared Mongoclient instance of the class

    database: Database
        The specific database instance that MongoDBClient connects to.

    Methods:
    ---------
    __init__(database_name:str) -> None
        Initializes the MongoDB connection using the given database name.
    """
    client = None
    
    def __init__(self, database_name:str = DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
            if mongo_db_url is None:
                raise Exception(f"Environment variable '{MONGODB_URL_KEY}' is not set")
            
            MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile = ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection successful")
            
        except Exception as e:
            raise CustomException(e,sys)