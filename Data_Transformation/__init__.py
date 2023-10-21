import os

import numpy as np 
import pandas as pd

from config import DataTransformationConfig
from params import get_preprocessor
from exception import CustomException
from logger import logging
from utils import save_object, create_schema




class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed.')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')


            # Target Column to be predicted
            target_column_name = self.data_transformation_config.target_column_name
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]


            create_schema(pd.DataFrame(input_feature_train_df), pd.DataFrame(target_feature_train_df))

            # Getting preprocessor object
            logging.info('Obtaining preprocessing object.')
            preprocessing_obj = get_preprocessor()
            logging.info(f'Preprocessor Created from `params.yaml` file: \n{preprocessing_obj}.')
            
            ## Transforming using preprocessor object
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets, Successfull !!")
            
            #concatenates side by side
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved.')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.error("Exception occured in the initiate_datatransformation.")
            logging.error(CustomException(e))
            raise e

if __name__ == '__main__':
    from Data_Ingestion import DataIngestion

    obj=DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)