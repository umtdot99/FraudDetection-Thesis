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

class PNN:
    """ 
        Class to represent the autoencoder and reflect the customizable pattern. 

        Args: 
            X_train (numpy array) : X data. 
            y_train (numpy array) : y data.
            imbalance_handler (sampler) : Imbalance handler object.
            kernel (str) : Type of the kernel.
            bandwidth (float): Size of the bandwidth. 
    """
    def __init__(self, X_train, y_train, imbalance_handler, kernel = None, bandwidth = 0.1):

        self.X_train = X_train
        self.y_train = y_train
        X_res, y_res = imbalance_handler.resampling(self.X_train, self.y_train)
    
        self.X_res = X_res
        self.y_res = y_res
        self.kernel = kernel
        self.bandwidth = bandwidth
        

    def kernel_functions(self):
        """ 
        Kernel functions to be used in PNN. 
        
            Returns: 
                Returns the function depending on the kernel type.
        """
        if self.kernel == "uniform": 
            
            return lambda x: 0.5 if np.abs(x/self.bandwidth) <= 1 else 0
            
            
        elif self.kernel == "triangular": 

            return lambda x: (1 - np.abs(x/self.bandwidth)) if np.abs(x/self.bandwidth) <= 1 else 0
           
            
        elif self.kernel == "epanechnikow": 

            return lambda x: 0.75*(1 - (x/self.bandwidth)**2) if np.abs(x/self.bandwidth) <= 1 else 0
            
            
        elif self.kernel == "gaussian": 

            return lambda x: (1/((2*np.pi)**0.5))*np.exp(-0.5*(x/self.bandwidth)**2)
           

            
    def pattern_layer(self, x):
        """ 
        Pattern layer to apply kernel values.
        
            Parameters: 
                x (array) : Training array of x.  
        
            Returns: 
                Returns values as array after kernel function implementation.
        """
        
        x_arr = np.array(x, dtype=float)
        euclidean_dist = np.linalg.norm(self.X_res - x_arr, axis = 1)
        values = []
        kernel_values = self.kernel_functions()
        for d in euclidean_dist: 
            values.append(kernel_values(d))

        values_arr = np.array(values)
        
        return values_arr 

    def summation_layer(self, values_arr):

        """ 
        Summation layer to calculate the average.
        
            Parameters: 
                values_arr (array) : Array from pattern layer.  
        
            Returns: 
                Returns values as list after summation.
        """
        y_arr = np.array(self.y_res)
        
        sum_values_0 = values_arr[y_arr==0].sum()
        sum_values_1 = values_arr[y_arr==1].sum()

        layer_values = [sum_values_0, sum_values_1]

        average_values_0 = sum_values_0 / (self.y_res.value_counts()[0])
        average_values_1 = sum_values_1 / (self.y_res.value_counts()[1])

        layer_val_avg = [average_values_0, average_values_1]
        return layer_val_avg

    def output_layer(self, layer_values):
        """ 
        Output layer to calculate the probabilities with considering prior probabilities (Applying Bayesian).
        
            Parameters: 
                layer_values (array) : List from summation layer.  
        
            Returns: 
                Returns the max value.
        """
        prior_prob0 = len(self.y_res[self.y_res==0])/len(self.y_res)
        prior_prob1 = len(self.y_res[self.y_res==1])/len(self.y_res)

        posterior_score0 = layer_values[0]*prior_prob0
        posterior_score1 = layer_values[1]*prior_prob1

        max_value = np.argmax([posterior_score0, posterior_score1])
        
        
        return max_value


    def pnn_pred(self, input_test):
        """ 
        Final layer to use each layer together.
        
            Parameters: 
                input_test (array) : Array for testing.  
        
            Returns: 
                Returns labels.
        """
        input_test_arr = np.array(input_test)
        labels = []
        for i in input_test_arr:

            k = self.pattern_layer(i)
            s = self.summation_layer(k)
            o = self.output_layer(s)

            labels.append(int(o))

        return labels

    def analysis(self, y_true, X_test):

        """ 
        Analysis to check the performance of PNN.
        
            Parameters: 
                y_true (array) : Array for true values of y.
                X_test (array) : Test array. 

            Returns: 
                Returns confusion matrix.
        """
        labels = self.pnn_pred(X_test)
        labels_arr = np.array(labels)
        
        cm = confusion_matrix(y_true, labels_arr)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual Class")
        plt.xlabel("Predicted Class")

        plt.show()
            
