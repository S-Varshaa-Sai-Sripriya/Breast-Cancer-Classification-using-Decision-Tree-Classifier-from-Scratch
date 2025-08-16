import numpy as np
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys


def accuracy(y_true, y_pred):
    try:
        logger.info("Calculating accuracy...")
        return np.sum(y_true == y_pred) / len(y_true)
    except Exception as e:
        logger.error("Error while calculating accuracy")
        raise CustomException(e, sys) from e
