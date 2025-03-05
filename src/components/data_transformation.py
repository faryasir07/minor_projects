##data clean ing  and feature engineering
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer    ##use to create pipeline
from sklearn.impute import SimpleImputer  
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:   ##input data transformation component
    preprocessing_obj_file_path=os.path.join("artifacts","preprocessing.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"  ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ("scaler",StandardScaler())
                ]
                
            )
            logging.info("numerical columns standarad scaling  is done!!")
            logging.info("categorical columns encoding is done!!")

            ##combining numerical and catoegorical pipeline
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e :
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
       try:
           train_data_df=pd.read_csv(train_path)
           test_data_df=pd.read_csv(test_path)
           logging.info("Training and Test Dataset is imported !!!")

           preprocessing_object=self.get_data_transformation_object()
           logging.info("preprocessing object obtained !!!")

           target_column_name="math_score"
           numerical_columns=["writing_score","reading_score"]

           input_features_train_df=train_data_df.drop(columns=[target_column_name],axis=1)
           target_feature_train_df=train_data_df[target_column_name]

           input_features_test_df=test_data_df.drop(columns=[target_column_name],axis=1)
           target_feature_test_df=test_data_df[target_column_name]

           logging.info("Applying preprocessing object on training and testing dataframe")

           input_features_train_df_arr=preprocessing_object.fit_transform(input_features_train_df)
           input_features_test_df_arr=preprocessing_object.transform(input_features_test_df)

           train_arr=np.c_[
               input_features_train_df_arr,np.array(target_feature_train_df)
           ]
           test_arr=np.c_[
               input_features_test_df_arr,np.array(target_feature_test_df)
           ]
           logging.info("Saved Preprocessing Objects...")
           save_object(
               file_path=self.data_transformation_config.preprocessing_obj_file_path,
               obj=preprocessing_object
            )

           return (train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessing_obj_file_path,
            
           )
       except Exception as e :
           raise CustomException(e,sys)

            