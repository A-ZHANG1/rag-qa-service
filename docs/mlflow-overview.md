# MLflow Overview

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides four main components:

## MLflow Tracking

MLflow Tracking allows you to log parameters, code versions, metrics, and artifacts when running your machine learning code. You can use it in any environment (standalone scripts, notebooks, etc.) to log results to local files or a remote server, then compare multiple runs.

Key concepts:
- **Runs**: Each execution of your code is a "run". You can log parameters, metrics, and artifacts to each run.
- **Experiments**: Runs are organized into experiments, which group related runs together.
- **Metrics**: Quantitative measures like accuracy, loss, AUC that you want to track over time.
- **Parameters**: Input configurations like learning rate, batch size, number of epochs.
- **Artifacts**: Output files like model weights, plots, data files.

## MLflow Models

MLflow Models is a convention for packaging machine learning models in multiple flavors, making it easy to deploy to diverse serving environments. Each MLflow Model is saved as a directory containing a `MLmodel` file that defines the flavors in which the model can be used.

Supported flavors include:
- Python Function (`python_function`)
- scikit-learn (`sklearn`)
- TensorFlow (`tensorflow`)
- PyTorch (`pytorch`)
- ONNX (`onnx`)
- Spark MLlib (`spark`)

## MLflow Model Registry

The Model Registry provides a centralized model store, set of APIs, and UI for collaboratively managing the full lifecycle of an MLflow Model. It provides:
- Model lineage (which run produced the model)
- Model versioning
- Stage transitions (e.g., from Staging to Production)
- Annotations and descriptions

## MLflow Projects

MLflow Projects are a standard format for packaging reusable data science code. Each project is simply a directory with code or a Git repository, and uses a descriptor file or convention to specify its dependencies and how to run the code.

## Getting Started

```python
import mlflow

# Start a run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 10)

    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)

    # Log model
    mlflow.sklearn.log_model(model, "model")
```

## MLflow in Microsoft Fabric

Microsoft Fabric integrates MLflow natively, providing a seamless experience for data scientists. Key features include:
- Autologging for popular frameworks
- Built-in experiment tracking UI
- Native model registry with workspace-level management
- Integration with Fabric Lakehouse for artifact storage
- Support for MLflow 3.0 with enhanced tracking capabilities
