import sys 
import os
 
from src.logger import logging

def error_message_detail(error : Exception, error_detail : sys) -> str:
    """
    Extracts detailed error information including file name , line no and error message.

    param: error: the exception that occured
    param: error_detail: the sys module to access traceback details.
    return: the detailed error message
    """
    _,_, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occured in python script name [{0}] and line number [{1}] error message [{2}]".format(file_name,exc_tb.tb_lineno, str(error))
    logging.info(error_message)
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)

        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message