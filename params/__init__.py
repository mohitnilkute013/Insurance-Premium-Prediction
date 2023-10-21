import yaml
import importlib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from logger import logging
from exception import CustomException

# Load the YAML parameters file
with open('params/params.yaml', 'r') as yaml_file:
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)


# Function to create an instance of a class
def create_instance(class_path, init_args=None):
    class_name = class_path.split('.')[-1]
    module_name = '.'.join(class_path.split('.')[:-1])
    
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    
    if init_args:
        return class_(**init_args)
    else:
        return class_()



def load_models() -> dict:
    """Loads all models specified in the params.yaml."""

    # Create a dictionary to store the model instances
    models = {}

    try:
        # Iterate through the model configurations
        for model_info in params['models']:
            model_name = model_info['name']
            model_class_path = model_info['class']
            
            if 'init_args' in model_info:
                init_args = model_info['init_args']
                
                if 'estimator' in init_args:
                    estimator_class_path = init_args['estimator']['class']
                    estimator_init_args = init_args['estimator'].get('init_args', {})
                    
                    # Create an instance of the estimator
                    estimator_instance = create_instance(estimator_class_path, estimator_init_args)
                    
                    # Update the init_args with the estimator instance
                    init_args['estimator'] = estimator_instance
                    
                # Create an instance of the model class with updated init_args
                model_instance = create_instance(model_class_path, init_args)
            else:
                # Create an instance of the model class with no init_args
                model_instance = create_instance(model_class_path)
            
            # Add the model instance to the dictionary
            models[model_name] = model_instance

    except Exception as e:
        logging.error(CustomException(e))
        raise e

    return models

    # Now, you have a dictionary of model instances
    # You can access them like this:
    # models['LinearRegression']
    # models['Lasso']
    # models['Ridge']
    # ... and so on



def load_tuning_models() -> dict:
    """Loads all tuned models specified in the params.yaml."""

    # Create a dictionary to store the tuned model instances
    tuned_models = {}

    try:
        # Iterate through the tuned model configurations
        for model_info in params['tune_models']:
            model_name = model_info['name']
            model_class_path = model_info['class']
            model_params = model_info['param_search_space']

            # Create an instance of the model class with no init_args
            model_instance = create_instance(model_class_path)

            # Add the tuned model instance to the dictionary
            tuned_models[model_name] = [model_instance, model_params]

    except Exception as e:
        logging.error(CustomException(e, sys))
        raise e

    return tuned_models



def get_preprocessor():

    """Makes a preprocessor object from the preprocessing steps in params.yaml"""

    # Initialize a dictionary to hold pipelines for each column group
    column_group_pipelines = {}

    try:
        # Iterate through column groups defined in the YAML file
        for column_group, preprocessing_steps in params['preprocessor']['preprocessing_steps'].items():
            # Initialize pipelines for the current column group
            pipelines = []
            
            # Iterate through preprocessing steps for the current column group
            for step_info in preprocessing_steps:
                step_name = step_info['name']
                step_class_path = step_info['class']
                
                if 'init_args' in step_info:
                    init_args = step_info['init_args']
                else:
                    init_args = None
                
                # Create an instance of the preprocessing step class
                preprocessing_step = create_instance(step_class_path, init_args)
                
                # Add the preprocessing step to the current pipeline
                pipelines.append((step_name, preprocessing_step))
            
            # Create a pipeline for the current column group
            column_group_pipelines[column_group] = Pipeline(pipelines)

        # Extract column groups
        column_groups = params['preprocessor']['column_groups']

        # Create a ColumnTransformer using the pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                (group, pipeline, column_groups[group]) for group, pipeline in column_group_pipelines.items()
            ]
        )

    except Exception as e:
        logging.error(CustomException(e,sys))
        raise e

    # Now, we have pipelines for each column group and a ColumnTransformer combining them
    return preprocessor