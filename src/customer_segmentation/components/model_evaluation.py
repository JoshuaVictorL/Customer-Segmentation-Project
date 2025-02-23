# Model Evaluation Pipeline with MLflow Integration

import pandas as pd
import joblib
import mlflow
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import json
from customer_segmentation.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted')
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
        return accuracy, precision, recall, f1

    def save_json(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_clusters = model.predict(test_x)
            
            accuracy, precision, recall, f1 = self.eval_metrics(test_y, predicted_clusters)

            # Save metrics locally
            scores = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            self.save_json(Path(self.config.evaluation_metrics_path), scores)

            # Log parameters from params.yaml and metrics to MLflow
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Model registry (if not file store)
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestSegmentModel")
            else:
                mlflow.sklearn.log_model(model, "model")
