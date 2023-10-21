from Data_Ingestion import DataIngestion
from Data_Transformation import DataTransformation
from Model_Trainer import ModelTrainer
from Model_Tuner import ModelTuner


def Train():
    
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer=ModelTrainer()
    success = model_trainer.initate_model_training(train_arr,test_arr)

    model_tuner = ModelTuner()
    success = model_tuner.initiate_model_tuning(train_arr,test_arr)

    return success

if __name__=='__main__':

    Train()
