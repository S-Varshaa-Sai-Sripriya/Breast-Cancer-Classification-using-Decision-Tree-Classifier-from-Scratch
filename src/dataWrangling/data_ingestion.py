import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from src.utils.logger import logger
from src.utils.exception import CustomException


class DataIngestion:
    def __init__(self):
        self.data = None

    def load_data(self):
        try:
            logger.info("Loading Breast Cancer dataset from sklearn...")
            dataset = load_breast_cancer()
            self.data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            self.data['target'] = dataset.target
            logger.info(f"Dataset loaded successfully with shape {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error("Error occurred while loading dataset")
            raise CustomException(e, sys) from e
