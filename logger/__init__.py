import os
import sys
import logging
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_S')}.log"
logs_path = os.path.join(os.getcwd(),'logs')
os.makedirs(logs_path,exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE_NAME)


logging_str = "[%(asctime)s] Line No. %(lineno)d %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    format = logging_str,
    level = logging.INFO,
    handlers = [
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)