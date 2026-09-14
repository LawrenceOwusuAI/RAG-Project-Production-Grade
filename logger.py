import logging
import os
from datetime import datetime

# LOG_FILE = f"{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.log"

# logs_path = os.path.join(os.getcwd(),"logs")
# os.makedirs(logs_path,exist_ok=True)
# LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

# logging.basicConfig(
#     filename=LOG_FILE_PATH,
#     format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
#     level=logging.INFO
# )
import logging
import os
import sys
from datetime import datetime


# =========================================================
# CREATE LOG DIRECTORY
# =========================================================

LOG_FILE = (
    f"{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.log"
)

logs_path = os.path.join(
    os.getcwd(),
    "logs",
)

os.makedirs(
    logs_path,
    exist_ok=True,
)

LOG_FILE_PATH = os.path.join(
    logs_path,
    LOG_FILE,
)


# =========================================================
# LOG FORMAT
# =========================================================

LOG_FORMAT = (
    "[ %(asctime)s ] "
    "%(lineno)d "
    "%(name)s - "
    "%(levelname)s - "
    "%(message)s"
)


formatter = logging.Formatter(
    LOG_FORMAT
)


# =========================================================
# FILE HANDLER
# =========================================================

file_handler = logging.FileHandler(
    LOG_FILE_PATH,
    encoding="utf-8",
)

file_handler.setFormatter(
    formatter
)


# =========================================================
# STDOUT HANDLER
# =========================================================

console_handler = logging.StreamHandler(
    sys.stdout
)

console_handler.setFormatter(
    formatter
)


# =========================================================
# ROOT LOGGER
# =========================================================

root_logger = logging.getLogger()

root_logger.setLevel(
    logging.INFO
)

root_logger.handlers.clear()

root_logger.addHandler(
    file_handler
)

root_logger.addHandler(
    console_handler
)