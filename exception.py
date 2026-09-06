from logger import logging
import sys

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_information=f"Error occured in python script name {file_name} line number {exc_tb.tb_lineno} and error message {str(error)}"

    return error_information

    

class CustomException(Exception):
    def __init__(self,error_information,error_detail:sys):
        super().__init__(error_information)
        self.error_information=error_message_detail(error_information,error_detail=error_detail)
    
    def __str__(self):
        return self.error_information


