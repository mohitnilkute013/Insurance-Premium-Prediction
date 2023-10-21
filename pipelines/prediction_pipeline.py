import sys
import os
from exception import CustomException
from logger import logging
from utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','tuned_model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)
            logging.info("Preprocessing Done.")

            pred=model.predict(data_scaled)
            logging.info(f"Prediction Completed -> Result : {pred[0]}")

            return pred
            

        except Exception as e:
            logging.error(CustomException(e))
            raise e
        
class CustomData:
    def __init__(self,
                 age:int,
                 sex:str,
                 bmi:float,
                 children:int,
                 smoker:str,
                 region:str):

        # Automatically set instance attributes based on constructor arguments
        
        for arg_name, arg_value in locals().items():
            if arg_name != 'self':
                setattr(self, arg_name, arg_value)

        # print(locals().items())
        # print(self.__dict__.items())


    def get_data_as_dataframe(self):
        try:
            
            df = pd.DataFrame([self.__dict__])
            logging.info(f'Dataframe Gathered\n {df.head().to_string()}')
            # print(self.__dict__.items())
            return df
        except Exception as e:
            logging.error('Exception Occured in Custom Data Gathering.')
            logging.error(CustomException(e))
            raise e


if __name__ == "__main__":

    cd = CustomData(age = 25,
                 sex = "male",
                 bmi = 29.2,
                 children = 1,
                 smoker = "yes",
                 region = "southwest")

    cd_df = cd.get_data_as_dataframe()

    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(cd_df)

    results = round(pred[0], 2)

    print(results)