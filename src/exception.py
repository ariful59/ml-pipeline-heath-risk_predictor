import sys


def error_messages(message):
    _, _, tb = sys.exc_info()
    filename = tb.tb_frame.f_code.co_filename
    lineno = tb.tb_lineno
    error_message = f"Error occured in python script name {filename} line number {lineno} error message {message}"

    return error_message

class Error(Exception):
    def __init__(self, error):
        super().__init__(error)
        self.error_details = error_messages(error)
        print(self.error_details)


# if __name__ == "__main__":
#     try:
#         a = 10 / 0
#     except ZeroDivisionError as e:
#         Error(e)