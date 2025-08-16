import os
import json
import pandas as pd
from src.dataWrangling.data_ingestion import DataIngestion
from src.dataWrangling.data_preprocessing import DataPreprocessing
from src.model.trainer import Trainer
from src.evaluation.metrics import accuracy
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys


def evaluate():
    try:
        logger.info("=== Evaluation Started ===")

        # 1. Load data
        df = DataIngestion().load_data()

        # 2. Preprocess & split
        preprocessor = DataPreprocessing(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data()

        # 3. Train model
        trainer = Trainer(X_train, y_train, X_test, y_test)
        model = trainer.train()

        # 4. Predict
        logger.info("Making predictions on test data...")
        y_pred = model.predict(X_test.values)

        # 5. Evaluate
        acc = accuracy(y_test.values, y_pred)
        logger.info(f"Evaluation complete with accuracy: {acc:.4f}")

        # Ensure artifacts directory
        os.makedirs("artifacts", exist_ok=True)

        # Save metrics
        metrics = {"accuracy": acc}
        with open("artifacts/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Console output
        print("\n Evaluation Complete!")
        print(f"Accuracy: {acc:.4f}")

        print("\nSample Predictions (first 10):")
        preds_df = pd.DataFrame({
            "Actual": y_test.values[:10],
            "Predicted": y_pred[:10]
        })
        print(preds_df)

    except Exception as e:
        logger.error("Evaluation failed")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    evaluate()
