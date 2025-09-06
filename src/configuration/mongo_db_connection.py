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
        Initializes the MongoDB connection  using the given database name.
    """

client = None

def __init__(self, database_name:str = DATABASE_NAME) -> None:

    try:
        if MongoDBClient.client is None:
            mongodb_url
    except Exception as e:
        raise CustomException(e,sys)