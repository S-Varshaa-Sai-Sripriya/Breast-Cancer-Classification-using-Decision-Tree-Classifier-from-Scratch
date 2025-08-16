import sys

class CustomException(Exception):
    """
    Custom Exception class to capture error messages with traceback details.
    """
    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_detail: sys) -> str:
        _, _, exc_tb = error_detail.exc_info()
        filename = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occurred in script: {filename}, line {line_number}, message: {error_message}"

    def __str__(self):
        return self.error_message
