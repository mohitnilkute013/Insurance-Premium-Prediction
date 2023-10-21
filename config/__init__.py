import yaml
from dataclasses import dataclass


# Load the YAML configuration file
with open('config/config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)


@dataclass
class DataIngestionConfig:
    train_data_path:str = config['split_data']['train_data_path']
    test_data_path:str = config['split_data']['test_data_path']
    raw_data_path:str = config['load_data']['raw_data_path']
    data_path:str = config['source_data']['data_path']
    data_url:str = config['source_data']['data_url']
    split_ratio:float = config['split_data']['split_ratio']


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = config['preprocessor_obj_path']
    target_column_name:str = config['source_data']['target_column']


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = config['model_path']


@dataclass
class ModelTunerConfig:
    tuned_model_file_path=config['tune_path']
