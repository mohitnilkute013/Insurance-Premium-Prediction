import os
import sys
import pickle
import yaml
import numpy as np 
import pandas as pd

from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
from typing import Dict, Any

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from exception import CustomException
from logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.error('Exception occured during model saving...')
        logging.error(CustomException(e))
        raise e


def create_schema(input_df, target_df):
    # Get column names and data types
    schema = {'COLUMNS': {'input_features': {}, 'target_feature': {}}}

    for column, dtype in input_df.dtypes.items():
        schema['COLUMNS']['input_features'][column] = dtype.name
    for column, dtype in target_df.dtypes.items():
        schema['COLUMNS']['target_feature'][column] = dtype.name

    # Write the schema dictionary to a YAML file
    with open('schema.yaml', 'w') as yaml_file:
        yaml.dump(schema, yaml_file, default_flow_style=False)


def train_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {'Model_Name':[], 'Model': [],'Train_Acc': [], 'R2_Score': [], 'RMSE': [], 'MAE': []}
        
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            mlflow.set_experiment(f"Expe_{model_name}")

            logging.info(f'Training on {model}')

            # Train model
            model.fit(X_train,y_train)

            # Training Accuracy
            train_acc = model.score(X_train, y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 scores for test data
            test_model_score = r2_score(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            mse = mean_squared_error(y_test, y_test_pred)
            rmse = np.sqrt(mse)
            logging.info(f'R2_Score: {test_model_score}')

            report['Model_Name'].append(model_name)
            report['Model'].append(model)
            report['Train_Acc'].append(train_acc*100)
            report['R2_Score'].append(test_model_score*100)
            report['RMSE'].append(rmse)
            report['MAE'].append(mae)


            with mlflow.start_run():
                mlflow.log_param('Model Name', model_name)
                # mlflow.log_model('Model', model)
                mlflow.log_metric('Train_Acc', train_acc*100)
                mlflow.log_metric('R2_Score', test_model_score*100)
                mlflow.log_metric('RMSE', rmse)
                mlflow.log_metric('MAE', mae)

                predictions = model.predict(X_train)
                signature = infer_signature(X_train, predictions)

                mlflow.sklearn.log_model(
                        model, model_name, registered_model_name=model_name, signature=signature
                    )

        return report

    except Exception as e:
        logging.error('Exception occured during model training')
        logging.error(CustomException(e))
        raise e



def select_top_models(report, metric, top_n=3):
    # Sort the models based on the specified metric
    sorted_models = sorted(
        report, key=lambda x: x[metric], reverse=True
    )
    # Select the top N models
    top_models = sorted_models[:top_n]
    return top_models



def tune_models(X_train, y_train, X_test, y_test, model_params):
    try:
        report = {'Model_Name':[], 'Model': [],'Train_Acc': [], 'R2_Score': [], 'RMSE': [], 'MAE': []}

        for i in range(len(model_params)):
            model = list(model_params.values())[i][0]
            params = list(model_params.values())[i][1]
            print(model, params)
            model_name = list(model_params.keys())[i]

            mlflow.set_experiment(f"Expe_Tune_{model_name}")

            logging.info(f'Tuning {model}')

            # Create a RandomizedSearchCV object
            search = RandomizedSearchCV(model, params, n_iter=100, scoring='r2', cv=5, n_jobs=-1, verbose=10)

            # Perform hyperparameter tuning
            search.fit(X_train, y_train)

            # Get the best hyperparameters and corresponding evaluation metrics
            best_params = search.best_params_
            best_score = search.best_score_

            logging.info(f'Best Param: {search.best_params_}')
            logging.info(f'Best Score: {search.best_score_}')

            model.set_params(**search.best_params_)

            # # Create a GridSearchCV object
            # grid_search=GridSearchCV(estimator=model, param_grid=params, cv=5, verbose=10)

            # # Perform hyperparameter tuning
            # grid_search.fit(X_train,y_train)

            # logging.info(f'Best Estimator: {grid_search.best_estimator_}')
            # logging.info(f'Best Param: {grid_search.best_params_}')

            # model.set_params(**grid_search.best_params_)

            # Train model
            model.fit(X_train,y_train)

            # Training Accuracy
            train_acc = model.score(X_train, y_train)

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for test data
            test_model_score = r2_score(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            mse = mean_squared_error(y_test, y_test_pred)
            rmse = np.sqrt(mse)
            logging.info(f'R2_Score: {test_model_score}')

            report['Model_Name'].append(model_name)
            report['Model'].append(model)
            report['Train_Acc'].append(train_acc*100)
            report['R2_Score'].append(test_model_score*100)
            report['RMSE'].append(rmse)
            report['MAE'].append(mae)

            with mlflow.start_run():
                mlflow.log_param('Model Name', model_name)
                # mlflow.log_model('Model', model)
                mlflow.log_metric('Train_Acc', train_acc*100)
                mlflow.log_metric('R2_Score', test_model_score*100)
                mlflow.log_metric('RMSE', rmse)
                mlflow.log_metric('MAE', mae)

                predictions = model.predict(X_train)
                signature = infer_signature(X_train, predictions)

                mlflow.sklearn.log_model(
                        model, model_name, registered_model_name=model_name, signature=signature
                    )

        return report

    except Exception as e:
        logging.error('Exception occured during model tuning')
        logging.error(CustomException(e))
        raise e

    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error('Exception Occured in load_object function utils')
        logging.error(CustomException(e))
        raise e

