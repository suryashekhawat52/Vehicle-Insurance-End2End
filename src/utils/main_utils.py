import os 
import sys

from pandas import DataFrame
import numpy as np
import dill
import yaml


from src.logger import logging
from src.exception import CustomException

# we define general functions that is used across project

def read_yaml(file_path:str)-> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e,sys)
    
def write_yaml(file_path:str, content:object, replace: bool = False)-> None:

    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as file:
                yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path:str)-> object:
    """Returns object/model from project directory
       file_path:str location of file to load
       return model/obj
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        return obj
    except Exception as e:
        raise CustomException(e,sys)
    
def save_objects(file_path:str, obj:object)-> object:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("file object saved")

    except Exception as e:
        raise CustomException(e,sys)
    
def save_numpy_array_data(file_path:str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys)

def load_numpy_array_data(file_path:str) -> np.array:
    """Load numpy array data from file"""
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

    



