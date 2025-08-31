# from src.logger import logging 

# logging.info("this is the info")

# 2 testing the custom exception class
import sys
from src.logger import logging
from src.exception import CustomException
try:
    a = 1 + '2'
except Exception as e:
    logging.info(e)
    raise CustomException(e,sys)