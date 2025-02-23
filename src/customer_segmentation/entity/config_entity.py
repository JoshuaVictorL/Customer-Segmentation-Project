from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    transformed_data_path: Path
    clustered_data_path: Path
    model_save_path: Path
    test_data_path: Path
    test_size: float
    random_state: int
    n_estimators: int
    max_depth: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: str
    model_path: str
    test_data_path: str
    evaluation_metrics_path: str
    mlflow_uri: str
    target_column: str
    all_params: dict