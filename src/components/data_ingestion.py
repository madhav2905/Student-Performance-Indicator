import os 
import sys 
import pandas as pd
from src.exception import CustomException 
from src.logger import logger 
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass 

@dataclass
class DataIngestionConfig: 
    raw_data_path: str=os.path.join('artifacts',"raw.csv")
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv") 

class DataIngestion: 
    def __init__(self):
        self.config=DataIngestionConfig() 
   
    def initiate_data_ingestion(self):
        logger.info("Entered in the data ingestion component")

        try: 
            df=pd.read_csv('notebooks/data/StudentsPerformance.csv')
            logger.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.config.raw_data_path),exist_ok=True)

            df.to_csv(self.config.raw_data_path,index=False)

            logger.info("Train Test split is initiated") 

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42) 

            train_set.to_csv(self.config.train_data_path,index=False)
            test_set.to_csv(self.config.test_data_path,index=False) 

            logger.info("Ingestion is completed")

            return (self.config.train_data_path,self.config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__": 
    ingestion=DataIngestion() 
    train_path,test_path=ingestion.initiate_data_ingestion()
    print(f"Train data path:{train_path}")
    print(f"Test data path:{test_path}")