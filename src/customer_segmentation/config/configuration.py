from customer_segmentation.constants import *
from customer_segmentation.utils.common import read_yaml, create_directories
from customer_segmentation.entity.config_entity import (DataIngestionConfig, DataValidationConfig, 
                                                        DataTransformationConfig, ModelTrainerConfig,
                                                        ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_trainer

        return ModelTrainerConfig(
            root_dir=config.root_dir,
            transformed_data_path=config.transformed_data_path,
            clustered_data_path=config.clustered_data_path,
            test_data_path=config.test_data_path,
            model_save_path=config.model_save_path,
            test_size=params.test_size,
            random_state=params.random_state,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth
        )
    

    def get_model_evaluation_config(self):
        config = self.config['model_evaluation']
        params = self.params['model_trainer']

        create_directories([config['root_dir']])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config['root_dir'],
            model_path=config['model_path'],
            test_data_path=config['test_data_path'],
            evaluation_metrics_path=config['evaluation_metrics_path'],
            mlflow_uri=config['mlflow_uri'],
            target_column=config['target_column'],
            all_params=params
        )

        return model_evaluation_config