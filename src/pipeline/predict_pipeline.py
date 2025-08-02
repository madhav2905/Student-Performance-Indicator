import os
import sys
import joblib
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")

            preprocessor=joblib.load(preprocessor_path)
            model=joblib.load(model_path)

            data_scaled=preprocessor.transform(features)
            prediction=model.predict(data_scaled)

            return prediction

        except Exception as e:
            raise CustomException(e,sys)