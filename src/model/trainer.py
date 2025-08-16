from .model import DecisionTreeClassifierScratch
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys


class Trainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train.values
        self.y_train = y_train.values
        self.X_test = X_test.values
        self.y_test = y_test.values
        self.model = DecisionTreeClassifierScratch(max_depth=10)

    def train(self):
        try:
            logger.info("Starting training process...")
            self.model.fit(self.X_train, self.y_train)
            logger.info("Training finished successfully")
            return self.model
        except Exception as e:
            logger.error("Training failed")
            raise CustomException(e, sys) from e
