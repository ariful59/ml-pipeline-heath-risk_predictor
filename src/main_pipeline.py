from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import Error
from src.logger import logging

if __name__ == "__main__":
    try:
        logging.info(">>>>>> Pipeline Started <<<<<<")
        
        # 1. Ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        # 2. Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        # 3. Training
        model_trainer = ModelTrainer()
        accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        print(f"Final Model Accuracy: {accuracy:.4f}")
        logging.info(">>>>>> Pipeline Completed Successfully <<<<<<")

    except Exception as e:
        print("Pipeline Failed")
        raise Error(e)
