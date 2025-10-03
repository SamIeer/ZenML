import numpy as np
import pandas as pd 
from zenml import pipeline, step 
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICES_START_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import mlflow_deploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BasePatameters, output

from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model

DockerSettings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """Deployment trigger config"""
    min_accuracy: float = 0

@step 
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    """Implements a simple model deployment trigger that looks at the Input model accuray"""
    return accuracy >= config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParaemters):
    pipeline_name: str
    step_name: str 
    running: bool = True 


@pipeline(enable_cache=False, settings=("docker" : docker_settings))
def continuous_deployment_pipeline(
    data_path: str
    min_accuracy: float = 0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_data(data_path = data_path)
    X_train, X_test, y_train, y_test = clean_data()
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model = model,
        deploy_decision = deployment_decision,
        workers = workers,
        timeout = timeout,
    )
