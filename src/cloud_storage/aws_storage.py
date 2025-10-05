import boto3 
import os 
from src.configuration.aws_connection import S3Client
from io import StringIO 
from typing import Union, List
import sys 
from src.logger import logging 
from src.exception import CustomException

from mypy_boto3_s3.service_resource import Bucket
from botocore.exceptions import ClientError
import pandas 
import pickle 

class SimpleStorageService:
    """
    A class for interacting with S3 Storage, providing methods for file management,
    data uploads and data retrieval in s3 buckets
    """
    
    def __init__(self):
        """
        Initializes the SimpleStorageService instance with S3 resource and client
        from S3Client class
        """
        s3_client = S3Client()
        self.s3_client = s3_client.s3_resource
        self.s3_resource = s3_client.s3_client 

    def s3_key_path_available(self, bucket_name, s3_key) -> bool:
        """
        Checks if a specified S3 key path (file path) is available in the specified bucket

        Args:
            bucket_name: Name of S3 bucket
            s3_key: Key path of file to check
        
        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            return len(file_objects)>0
        except Exception as e:
            raise CustomException(e,sys)
    
    @staticmethod
    def read_object(object_name: str, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str]:
        """
        Reads the specified S3 object with optional decoding and formatting.

        Args:
            object_name (str): The S3 object name.
            decode (bool): Whether to decode the object content as a string.
            make_readable (bool): Whether to convert content to StringIO for DataFrame usage.

        Returns:
            Union[StringIO, str]: The content of the object, as a StringIO or decoded string.
        """
        # logging.info("Entered the read_object method of SimpleStorageService class")
        try:
            # Read and decode the object content if decode=True
            func = (
                lambda: object_name.get()["Body"].read().decode()
                if decode else object_name.get()["Body"].read()
            )
            # Convert to StringIO if make_readable=True
            conv_func = lambda: StringIO(func()) if make_readable else func()
            # logging.info("Exited the read_object method of SimpleStorageService class")
            return conv_func()
        except Exception as e:
            raise CustomException(e, sys)
