from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=="__main__": 
    ingestion=DataIngestion() 
    train_path,test_path=ingestion.initiate_data_ingestion() 
    
    transformer=DataTransformation() 
    X_train_transformed,X_test_transformed,y_train,y_test,preprocessor_path=transformer.initiate_data_transformation(train_path,test_path) 
    print("Data pipeline executed successfully.")
    print("Preprocessor saved at:",preprocessor_path)

    trainer=ModelTrainer()
    metrics=trainer.train_model(X_train_transformed,X_test_transformed,y_train,y_test)
    print("Model training completed")
    print(f"MSE:{metrics["mse"]:.4f}")
    print(f"MAE:{metrics["mae"]:.4f}")
    print(f"R2 Score:{metrics["r2_score"]:.4f}")