import sys 
from src.logger import logger

def get_error_details(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    filename=exc_tb.tb_frame.f_code.co_filename
    line_number=exc_tb.tb_lineno
    return f"Error occurred in script: [{filename}] at line [{line_number}] with error: {str(error)}"

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=get_error_details(error_message,error_detail)
    
    def __str__(self):
        return self.error_message         