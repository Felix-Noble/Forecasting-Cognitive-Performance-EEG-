#set_up_loggers.py
import logging
import os
import sys

def make_logger(name=None, level_console=logging.DEBUG, level_file=logging.ERROR):
    logger = logging.getLogger(name)
    logger.setLevel(level_console)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(os.path.join("logs", f"{name}.log"))
    file_handler.setLevel(level_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level_console) # Log everything to the console for development.
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger