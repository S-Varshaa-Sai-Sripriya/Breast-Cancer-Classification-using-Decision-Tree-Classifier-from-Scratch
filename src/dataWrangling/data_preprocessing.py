import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys


class DataPreprocessing:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        try:
            logger.info("Splitting dataset into train and test sets...")
            X = self.data.drop(columns=["target"])
            y = self.data["target"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            logger.info(f"Data split completed: Train={X_train.shape}, Test={X_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error("Error occurred during data splitting")
            raise CustomException(e, sys) from e
