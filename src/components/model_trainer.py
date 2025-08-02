import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException 
from src.logger import logger 
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

@dataclass
class ModelTrainerConfig: 
    trained_model_file_path: str=os.path.join('artifacts',"model.pkl") 

class ModelTrainer:
    def __init__(self):
        self.config=ModelTrainerConfig()
    
    def train_model(self,X_train,X_test,y_train,y_test): 
        try: 
            logger.info("Starting model training process")

            model=LinearRegression()
            model.fit(X_train,y_train) 

            logger.info("Model training completed") 

            y_pred=model.predict(X_test)

            mse=mean_squared_error(y_test,y_pred)
            mae=mean_absolute_error(y_test,y_pred)
            r2=r2_score(y_test,y_pred) 

            logger.info(f"Model Evaluation Metrics:\nR2 Score:{r2}\nMSE:{mse}\nMAE:{mae}")

            joblib.dump(model,self.config.trained_model_file_path)
            logger.info("Trained model saved successfully")

            return {"mse":mse,"mae":mae,"r2_score":r2,"model_path":self.config.trained_model_file_path}


        
        except Exception as e:
            raise CustomException(e,sys)