#!/usr/bin/env python
# coding: utf-8


import pickle
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer, FunctionTransformer, PolynomialFeatures
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, auc, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
import import_ipynb
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.silhouette import silhouette
import shap
import umap
from xgboost import XGBClassifier #https://xgboost.readthedocs.io/en/release_3.0.0/get_started.html
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class DataGathering: 
    
    """ 
        Class to represent the pre-processing and data transformation steps. 

        Parameters: 
            file_path (str) : Path of the CSV file. 
            target (str) : Target of the data. 
            scaler (str) : Scaler for standardization or normalization of data. 
            imputer (str) : Imputer for missing value treatment. 
            pipeline_numerical (Pipeline): Pipeline object for pre-processing on numerical columns.
            pipeline_categorical (Pipeline): Pipeline object for pre-processing on categorical columns.
            pipeline_bin (Pipeline): Pipeline object for pre-processing on bin columns.
            pipeline_num_list (list[str]): List that includes pre-processing steps as lists.
            pipeline_cat_list (list[str]): List that includes pre-processing steps as lists.
            bin_list (lst[str]): List that includes pre-processing steps as lists.
            autoencoder (str): String to indicate whether autoencoders are going to be used or not. 
            dict_map (dict): Dictionary to convert the target into integers. 
            separator (str): Separator of the CSV file.
            calibration_size (float): Size of the calibration size.
            test_size (float): Size of the test set. 
            val_size (float): Size of the val set.
            data_type (str): Type of the data for neural networks to avoid confusion.
            batch_size (int): Size of the batch for data loaders in neural network. 
            
        
    """
    def __init__(self, file_path, calibration_size = None, test_size=0.2, val_size = 0.3, batch_size = None, 
                 pipeline_num_list=None,  pipeline_cat_list=None, dict_map =None, target=None, data_type = None, pipeline_numerical=None, 
                 pipeline_categorical=None, pipeline_bin = None, bin_list = None, autoencoder = None, separator = None):
        
        self.file_path = file_path
        self.target = target
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy = "median")
        self.pipeline_numerical = pipeline_numerical 
        self.pipeline_categorical = pipeline_categorical
        self.pipeline_bin = pipeline_bin
        self.pipeline_num_list = pipeline_num_list
        self.pipeline_cat_list = pipeline_cat_list 
        self.autoencoder = autoencoder
        self.dict_map = dict_map
        self.separator = separator
        self.bin_list = bin_list
        self.calibration_size = calibration_size
        self.test_size = test_size
        self.val_size = val_size
        self.data_type = data_type
        self.batch_size = batch_size
    
    def get_preprocessing_step(self, method): 
        """ 
        Class to represent the pre-processing and data transformation steps. 

        Parameters: 
            method (str): Pre-processing method.

        Returns: 
            Returns the type of the pre-processor.
            
        
    """
        if method == "imputer_num": 
            return SimpleImputer(strategy = "median")

        elif method == "scaler" :
            return StandardScaler()

        elif method == "imputer_cat": 
            return SimpleImputer(strategy="most_frequent")

        elif method == "onehot":
            return OneHotEncoder()

        else: 
            raise ValueError("Please enter a valid response !")
            
            
    def get_data_objects(self):
        
        """
            Preprocessing step that is identifed by pipelines and column transformer. 

            Returns: 
                preprocessor (transformer): Preprocessor to transform the data into an appropriate format. 
        """

        df_ = pd.read_csv(self.file_path, encoding="utf-8-sig", sep= self.separator, skipinitialspace=True)  


        df = df_.drop_duplicates().reset_index(drop=True)
        
        if self.target is not None:
            
            X = df.drop(columns=[self.target])
            if self.dict_map is not None: 
                y = df[self.target].astype(str).str.strip().str.lower().map(self.dict_map).astype("int8")
            else: 
                y = df[self.target].astype(str).str.strip().str.lower().astype("int8")
                
            bins = self.bin_list
            numerical_columns = X.select_dtypes(include=['number']).columns.to_list()
            categorical_columns = X.select_dtypes(include=['object']).columns.to_list()

        else: 
            X = df
            numerical_columns = X.select_dtypes(include=['number']).columns.to_list()
            categorical_columns = X.select_dtypes(include=['object']).columns.to_list()
            bins = self.bin_list

        steps_num = []
        steps_cat = []
        if (self.pipeline_num_list is not None) or (self.pipeline_cat_list is not None):
            for i in range(len(self.pipeline_num_list)):
                steps_num.append((f"{self.pipeline_num_list[i]}", self.get_preprocessing_step(self.pipeline_num_list[i])))
                self.pipeline_numerical = Pipeline(steps=steps_num)

            for j in range((len(self.pipeline_cat_list))):
                steps_cat.append((f"{self.pipeline_cat_list[j]}",self.get_preprocessing_step(self.pipeline_cat_list[j])))
                self.pipeline_categorical = Pipeline(steps=steps_cat)
                
        if self.bin_list is not None:
            preprocessor = ColumnTransformer(
                [
                    ('pipeline_num',self.pipeline_numerical, numerical_columns),
                    ('pipeline_bin', self.pipeline_bin, bins),
                    ('pipeline_cat',self.pipeline_categorical, categorical_columns)
                ]
            )

        else: 
            preprocessor = ColumnTransformer(
                [
                    ('pipeline_num',self.pipeline_numerical, numerical_columns),
                    ('pipeline_cat',self.pipeline_categorical, categorical_columns)
                ]
            )
     
        print(preprocessor)
        return preprocessor
         

        
    def main_split(self):

        """
            Applies the preprocessor and concatanates the X and y to create the training and test data.

            Returns: 
                X_train_df (data frame): Data frame for training.
                X_test_df (data frame): Data frame for testing.
                y_train (data series): Data series for training in case of supervised learning.
                y_test (data series): Data series for testing in case of supervised learning.
        """
        
        df_ = pd.read_csv(self.file_path, encoding="utf-8-sig", sep= self.separator, skipinitialspace=True)  

        df = df_.drop_duplicates().reset_index(drop=True)
            
        if self.target is not None:
                
            X = df.drop(columns=[self.target])
            if self.dict_map is not None: 
                y = df[self.target].astype(str).str.strip().str.lower().map(self.dict_map).astype("int8")
            else: 
                y = df[self.target].astype(str).str.strip().str.lower().astype("int8")
    
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = self.test_size)
    
            X_train_df = X_train.reset_index(drop = True)
            X_test_df = X_test.reset_index(drop = True)
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

                
            return X_train_df, y_train, X_test_df, y_test
        
        else: 
            X = df
            X_train, X_test = train_test_split(X, test_size = self.test_size)
            
            X_train_df = X_train.reset_index(drop = True)
            X_test_df = X_test.reset_index(drop = True)
          

            return X_train_df, X_test_df


    def calibration_loading(self, X_train, y_train):
        """
            Creating the calibration set. 

            Parameters: 
                X_train (data frame): Training data set X.
                y_train (data series): Training data set y.
                
            Returns: 
                X_train_ (data frame): Data frame for training.
                X_calib (data frame): Data frame for calibration.
                y_train_ (data series): Data series for training.
                y_calib (data series): Data series for calibration.
        """
        if self.calibration_size is not None:
                
            X_train_, X_calib, y_train_, y_calib = train_test_split(X_train, y_train, stratify = y_train, test_size = self.calibration_size)
            
            
            return X_train_, y_train_, X_calib, y_calib

        else: 
            pass
    
    def data_loading(self, imbalance_handler = "none"):

        """
            Prepares a data loader for neural networks. 

            Parameters: 
                imbalance_handler (str): Type of imbalance handler
                
            Returns: 
                data_loader_train (data loader): Data loader for training.
                data_loader_validation (data loader): Data loader for validation. 
                data_loader_test (data loader): Data loader for testing. 
        """
        
        df_ = pd.read_csv(self.file_path, encoding="utf-8-sig", sep= self.separator, skipinitialspace=True)  

        df = df_.drop_duplicates().reset_index(drop=True)
        
        X = df.drop(columns=[self.target])
        if self.dict_map is not None: 
                y = df[self.target].astype(str).str.strip().str.lower().map(self.dict_map).astype("int8")
        else: 
            y = df[self.target].astype(str).str.strip().str.lower().astype("int8")

        X_train_a, X_valid_a, y_train, y_valid_a = train_test_split(X, y, stratify = y, test_size = self.test_size)
        X_valid_b, X_test_b, y_valid, y_test = train_test_split(X_valid_a, y_valid_a, test_size = self.val_size)
        
        preprocessor = self.get_data_objects()
        X_train_p = preprocessor.fit_transform(X_train_a)
        X_valid_p = preprocessor.transform(X_valid_b)
        X_test_p = preprocessor.transform(X_test_b)

        if imbalance_handler.method == "classweights": 

            X_res = X_train_p
            y_res = y_train
            
        else: 
            
            X_res, y_res = imbalance_handler.resampling(X_train_p, y_train)
        
        columns = preprocessor.get_feature_names_out()
        X_train = pd.DataFrame(X_res, columns = columns).reset_index(drop=True)
        X_valid = pd.DataFrame(X_valid_p, columns = columns).reset_index(drop=True)
        X_test = pd.DataFrame(X_test_p, columns = columns).reset_index(drop=True)
        y_res = y_res.reset_index(drop=True)
        
        if self.autoencoder == "yes":
            
            X_train_ae = X_train[y_res==0]
            y_train_ae = y_res[y_res==0]

    
            X_tensor = torch.tensor(X_train_ae.values, dtype = torch.float32)
            y_tensor = torch.tensor(y_train_ae.values, dtype = self.data_type)
        
        else: 
            
            X_tensor = torch.tensor(X_train.values, dtype = torch.float32)
            y_tensor = torch.tensor(y_res.values, dtype = self.data_type)

    
        X_valid_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
        y_valid_tensor = torch.tensor(y_valid.values, dtype=self.data_type)
        X_test_tensor = torch.tensor(X_test.values, dtype = torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype = self.data_type)
        

        tensor_data = TensorDataset(X_tensor, y_tensor)
        tensor_valid_data = TensorDataset(X_valid_tensor, y_valid_tensor)
        tensor_test_data = TensorDataset(X_test_tensor, y_test_tensor)
        
        data_loader_train = DataLoader(tensor_data, batch_size = self.batch_size, shuffle = True)
        data_loader_valid = DataLoader(tensor_valid_data, batch_size=self.batch_size)
        data_loader_test = DataLoader(tensor_test_data, batch_size = self.batch_size)
        
        return data_loader_train, data_loader_valid, data_loader_test 
        
