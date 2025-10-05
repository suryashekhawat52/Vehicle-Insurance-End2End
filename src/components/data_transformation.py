import os 
import sys 
import pandas as pd
import numpy as np 
import re

from imblearn.combine import SMOTEENN 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder , TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from src.exception import CustomException
from src.logger import logging 
from src.utils.main_utils import save_objects, read_yaml, save_numpy_array_data

class FeatureNamesCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        self.preprocessor.fit(X,y)
        raw_names = self.preprocessor.get_feature_names_out()
        #cleaner names
        clean_names =  [re.sub(r'[^0-9a-zA-Z_]+','_',name) for name in raw_names]
        self.feature_names_out_ = clean_names
        return self
    
    def transform(self, X):
        Xt = self.preprocessor.transform(X)
        return pd.DataFrame(Xt, columns=self.feature_names_out_)
    
    def get_features_names_out(self):
        return self.feature_names_out_
    


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data,
        including gender mapping and dummy variable creation, column renaming
        feature scaling and type adjustments.
        """
        logging.info("Entered get data_transformer_object")

        try:
            logging.info("Transformer Initialized: Standard scaler and minmax scaler")

            # load schema configuration
            std_columns = self._schema_config['std_columns']
            mm_columns = self._schema_config['mm_columns']
            ohe_columns = self._schema_config['ohe_columns']
            target_encode_columns = self._schema_config['target_encode_columns']


            logging.info("Columns loaded from schema")

            #creating preprocessor pipelines for each columns defined
            std_pipeline = Pipeline([
                  ("imputer", SimpleImputer(strategy='median')),
                  ("scaler",StandardScaler())
                  ])
            
            mm_pipeline = Pipeline([
                  ("imputer", SimpleImputer(strategy='median')),
                  ("scaler",MinMaxScaler())
                  ])
            target_pipeline = Pipeline([
                  ("imputer",SimpleImputer(strategy="most_frequent")),
                  ("target_enc",TargetEncoder(smooth='auto'))
                  ])
            
            ohe_pipeline = Pipeline([
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ("ohe",OneHotEncoder(drop = 'first',handle_unknown='ignore'))
                ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("std",std_pipeline, std_columns),
                    ("mm", mm_pipeline, mm_columns),
                    ("ohe",ohe_pipeline, ohe_columns),
                    ("te", target_pipeline, target_encode_columns)
                ],
                remainder='passthrough'
                
            )

            cleaner_preprocessor = FeatureNamesCleaner(preprocessor=preprocessor)

            logging.info("Final pipeline ready")
            logging.info("Exited get_data_transformer_object method of DataTransformation Class")
            return cleaner_preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """Initiates the data transformation component for the pipeline"""

        try:
            logging.info("Data Transformation started")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train Test data loaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis = 1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis = 1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("input and target columns defined for each train and test set")
            
            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")
            

            input_feature_train_df = pd.DataFrame(preprocessor.fit_transform(input_feature_train_df,target_feature_train_df),columns = preprocessor.get_features_names_out())
            input_feature_test_df = pd.DataFrame(preprocessor.transform(input_feature_test_df),columns = preprocessor.get_features_names_out())
            print(input_feature_train_df.head())
            logging.info("Applying SMOTEENN to handle imbalanced dataset")
            smt = SMOTEENN(sampling_strategy='minority')
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_df,target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_df,target_feature_test_df
            )
            logging.info("SMOTEENN applied to train test df")

            train_arr = np.c_[input_feature_train_final,np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final,np.array(target_feature_test_final)]
            logging.info("Input and target array concatenation done for train and test df")

            save_objects(self.data_transformation_config.transformed_object_file_path,preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            logging.info("Saving transformation object and transformed files")

            logging.info("Data Transformation completed successfully")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact
        
        except Exception as e:
            raise CustomException(e,sys)




