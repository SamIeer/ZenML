from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configs"""
    model__name: str =  "LinearRegression"
