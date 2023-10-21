import os
import pandas as pd

from logger import logging
from exception import CustomException
from config import DataIngestionConfig

from sklearn.model_selection import train_test_split


## Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts.')
        try:
            df=pd.read_csv(self.ingestion_config.data_path)
            logging.info('Dataset read as pandas Dataframe.')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            split_ratio = self.ingestion_config.split_ratio
            train_set,test_set=train_test_split(df,test_size=split_ratio,random_state=42)
            logging.info('Train test split done with 30% test size.')

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data saved as Train and Test in artifacts folder.")

            logging.info('Ingestion of Data is completed.')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
            
        except Exception as e:
            logging.error(CustomException(e))
            raise e



if __name__=='__main__':
    data_injestor = DataIngestion()
    data_injestor.initiate_data_ingestion()