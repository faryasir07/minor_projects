import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

##for every component we create congfig file
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and testing data  !!!")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1] , ##last one ko test main lena hai isiliye
                train_array[:,-1], ##aall row values of last coulmns
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Splited training and testing data  !!!")

            logging.info("Creating dictionary of all models !!!")
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbours":KNeighborsRegressor(),
                "xgbregressor":XGBRegressor(),
                "CatBoosting":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()
            }
            logging.info("Created dictionary of all models !!!")
            logging.info("Hyper parameter tuning initiated !!!")

            params={
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Linear Regression": {},

                "K-Neighbours": {
                   'n_neighbors': [3, 5, 7],  
                   'weights': ['uniform'],
                   'algorithm': ['auto']
                },
                "xgbregressor": {
                     'learning_rate': [ 0.1, 0.2],
                     'n_estimators': [50, 100],
                     'max_depth': [3, 5]
                     #'min_child_weight': [1, 3, 5],
                     #'gamma': [0, 0.1, 0.2, 0.3],
                     #'subsample': [0.7, 0.8, 0.9, 1.0],
                     #'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
                       },
                "CatBoosting":{
                    'depth': [6,8,10],
                    'learning_rate': [0.1],
                    'iterations': [30, 50]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }  
                
            }
            logging.info("Creating a function  for model Evaluation !!! ")
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)


            ##to get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            ##to get best model name from dict
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best moidel found")
            
            logging.info("Best Model Found on both training and test datasets!!!")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)

            return r2_square

        except Exception as e:
            raise CustomException (e,sys)





