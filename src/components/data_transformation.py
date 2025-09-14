import os 
import sys 
import pandas as pd
import numpy as np 

from imblearn.combine import SMOTEENN 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from src.exception import CustomException
from src.logger import logging 
from src.utils.main_utils import save_objects, read_yaml, save_numpy_array_data

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
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info("Transformer Initialized: Standard scaler and minmax scaler")

            # load schema configuration
            num_features = self._schema_config['num_features']
            mm_features = self._schema_config['mm_columns']
            logging.info("Cols loaded from schema")

            #creating preprocessor pipeline
            preprocessor = ColumnTransformer(transformers=[
                ("StandardScaler",numeric_transformer,num_features),
                ("MinMaxScaler",min_max_scaler,mm_features)
            ],
            remainder="passthrough"
            )
            final_pipeline = Pipeline(steps=[("preprocessor",preprocessor)])
            logging.info("Final pipeline ready")
            logging.info("Exited get_data_transformer_object method of DataTransformation Class")
            return final_pipeline
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def _map_gender_column(self, df):
        """Map gender column to 0 for Female and 1 for Male"""
        logging.info("Mapping Gender column to binary values")
        df['Gender'] = df['Gender'].map({'Female':0,'Male':1}).astype('int')
        return df 
    
    def _create_dummy_columns(self,df):
        """
        create dummy variables for categorical features
        """
        logging.info("Creating dummy variables for cat features")
        df = pd.get_dummies(df, drop_first=True)
        return df
    
    def _rename_columns(self,df):
        """Renaming columns and ensure integer type for """
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns = {
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Year": "Vehicle_Age_gt_2_Year"
        })
        for col in ["Vehicle_Age_lt_1_Year","Vehicle_Age_gt_2_Year","Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df
    
    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists"""
        logging.info("Dropping the id column")
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(drop_col, axis = 1)

        return df
    
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

            input_feature_train_df = self._map_gender_column(input_feature_train_df)
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
            input_feature_train_df = self._rename_columns(input_feature_train_df)

            input_feature_test_df = self._map_gender_column(input_feature_test_df)
            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
            input_feature_test_df = self._rename_columns(input_feature_test_df)
            logging.info("Custom transformation applied to train and test df")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing the transformation on training data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing the transformation on testing data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Applying SMOTEENN to handle imbalanced dataset")
            smt = SMOTEENN(sampling_strategy='minority')
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr,target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr,target_feature_test_df
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




