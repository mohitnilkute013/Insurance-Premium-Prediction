import os
import numpy as np
import pandas as pd

from config import ModelTunerConfig
from params import load_tuning_models
from exception import CustomException
from logger import logging
from utils import save_object, tune_models


class ModelTuner:
    def __init__(self):
        self.tune_config = ModelTunerConfig()

    def initiate_model_tuning(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data.')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            # Load the required models from the params.yaml file
            logging.info("Loading all Model Tuning Dependencies and creating their instances according to `params.yaml` file.")
            model_params = load_tuning_models()

            model_report:dict = tune_models(X_train, y_train, X_test, y_test, model_params)
            print(pd.DataFrame(model_report))
            print('\n====================================================================================\n')
            logging.info(f'Model Report : \n{pd.DataFrame(model_report)}')

            # To get best model score from dictionary 
            index = list(model_report['R2_Score']).index(max(model_report['R2_Score']))

            best_model_name = model_report['Model_Name'][index]
            best_model = model_report['Model'][index]
            best_model_score = model_report['R2_Score'][index]
            best_model_train_acc = model_report['Train_Acc'][index]
            best_model_rmse = model_report['RMSE'][index]
            best_model_mae = model_report['MAE'][index]


            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}.')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}, Train Accuracy : {best_model_train_acc}, RMSE : {best_model_rmse}, MAE : {best_model_mae}.')

            save_object(
                 file_path=self.tune_config.tuned_model_file_path,
                 obj=best_model
            )

            logging.info('Saved Best Tuned Model file.')
            logging.info('Model Tuning Completed.  :)')


        except Exception as e:
            logging.error(CustomException(e))
            raise e

        return True


if __name__=='__main__':
    from Data_Ingestion import DataIngestion
    from Data_Transformation import DataTransformation
    from Model_Trainer import ModelTrainer

    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)

    model_tuner = ModelTuner()
    model_tuner.initiate_model_tuning(train_arr,test_arr)