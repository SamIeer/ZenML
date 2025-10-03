import logging 
from typing import Tuple

import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from typing_extensions import Annotated

from src.evaluation import MSE, R2 , RMSE

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker 
@step(experiment_tracker= experiment_tracker)
def evaluate_model(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    ) -> Tuple[
                    Annotated[float, "r2_score"],
                    Annotated[float, "rmse"]
]:
    """
    Evaluate the model on the ingested data

    Args:
        df (pd.DataFrame): the ingested data
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metrics("mse", mse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metrics("r2", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metrics("rmse", rmse)

        return r2 , rmse 
    except Exception as e:
        logging.error("Error in evaluating model: {}". format(e))
        raise e
    
# Compare this snippet from steps/model_train.py: