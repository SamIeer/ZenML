import logging 
import pandas as pd
from zenml import step

@step
def evaluate_model(df: pd.DataFrame) -> None:
    """
    Evaluate the model on the ingested data

    Args:
        df (pd.DataFrame): the ingested data
    """