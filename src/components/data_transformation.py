import os 
import sys 
import numpy as np
import pandas as pd 
from src.exception import CustomException 
from src.logger import logger 
from dataclasses import dataclass 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.impute import SimpleImputer
import joblib

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation: 
    def __init__(self):
        self.config=DataTransformationConfig()

    def get_preprocessing_pipeline(self): 
        try: 
            categorical_columns=['gender','race/ethnicity','parental level of education','lunch','test preparation course']
            numerical_columns=['reading score','writing score'] 

            categorical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('onehotencoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            numerical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer(
                transformers=[
                    ('categorical',categorical_pipeline,categorical_columns),
                    ('numerical',numerical_pipeline,numerical_columns)
                ]
            )

            logger.info("Preprocessor created successfully")

            return preprocessor
        except Exception as e: 
            raise CustomException(e,sys) 
        
    def initiate_data_transformation(self,train_path,test_path): 
        try:
            logger.info("Starting data transformation step")
            
            df_train=pd.read_csv(train_path)
            df_test=pd.read_csv(test_path)

            target_column='math score'

            X_train=df_train.drop(columns=[target_column])
            y_train=df_train[target_column]

            X_test=df_test.drop(columns=[target_column])
            y_test=df_test[target_column] 

            preprocessor=self.get_preprocessing_pipeline()

            X_train_transformed=preprocessor.fit_transform(X_train)
            X_test_transformed=preprocessor.transform(X_test)

            logger.info("Data has been transformed")

            joblib.dump(preprocessor,self.config.preprocessor_obj_file_path)
            logger.info("Saved preprocessing object") 

            return (X_train_transformed,X_test_transformed,y_train,y_test,self.config.preprocessor_obj_file_path)


        except Exception as e: 
            raise CustomException(e,sys)