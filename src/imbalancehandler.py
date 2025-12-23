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


from generator import Generator
from discriminator import Discriminator
from gan_ import Gan

class ImbalanceHandler:
    """ 
        Class to represent the autoencoder and reflect the customizable pattern. 

        Args: 
            method (str) : Name of the class imbalance handler technique.
            sampler (sampler object) : Sampler object.
            class_weights (dict) : Class weights to identify class imbalance.
            gan (gan object) : Initiating the GAN object.
            gan_epochs (int) : Epochs for GAN.
            gan_noise (int): Noise for the generator. 
            columns (list): Column names for new samples.
             
    """
    def __init__(self, method = "none", sampler = None, class_weights = None, gan=None, gan_epochs=None, gan_noise=None, columns=None):
        
        self.method = method
        self.sampler = sampler
        self.class_weights = class_weights
        self.gan = gan
        self.gan_epochs= gan_epochs
        self.gan_noise = gan_noise
        self.columns = columns
        
    def resampling(self, X, y):
        """ 
        Resampling method to initate the resampling of X and y.
        
            Parameters: 
                X (data frame) : Data frame of X.
                y (data series): Data series of y
        
            Returns: 
                Returns the sampled X and y.
        """
        
        if self.method == "none":
            return X, y
        
        elif self.method == "oversampling":
            self.sampler = RandomOverSampler()
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            return X_resampled, y_resampled

        elif self.method == "undersampling":
            self.sampler = RandomUnderSampler()
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            return X_resampled, y_resampled

        elif self.method == "nearmiss":
            self.sampler = NearMiss(version = 1)
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            return X_resampled, y_resampled

        elif self.method == "smote":
            self.sampler = SMOTE()
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            return X_resampled, y_resampled

        elif self.method == "gan": 

            
            generator = Generator(noise=64, output=30, hidden = 64)
            discriminator = Discriminator(input_dim = 30, hidden = 64)
            self.gan = Gan(generator, discriminator, lr_g =0.01, lr_d = 0.01)
            
            fraud = len(X[(y==1)])
            non_fraud = len(X[(y==0)])
            need = non_fraud - fraud

            fraud_values = X[(y==1)]
            tensor_x = torch.tensor(fraud_values, dtype=torch.float32)
            tensor_y = torch.tensor(y[(y==1)].to_numpy(), dtype=torch.float32)

            tensor_data = TensorDataset(tensor_x, tensor_y)
            dataloader = DataLoader(tensor_data, batch_size = 64, shuffle = True)

            self.gan.train(dataloader)
        
            synthetic_samples = self.gan.creating_new(need)

            X = pd.DataFrame(X, columns=self.columns)
            y = pd.Series(y).reset_index(drop=True)
            synth_data = pd.DataFrame(synthetic_samples, columns = X.columns)

            synth_data_y = pd.Series([1]*need)
            X_resampled = pd.concat([X, synth_data], ignore_index=True)
            y_resampled = pd.concat([y, synth_data_y], ignore_index=True)
            

            return X_resampled, y_resampled
            
        
        elif self.method == "classweights": #https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/

            if self.class_weights is None:
        
                weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y), y = y)
                self.class_weights = dict(zip(np.unique(y), weights))
            
            
            return X, y
            
        else:
            raise ValueError(f"{self.method} is not a valid method!")

    def sampler_(self, y):
        """ 
        Sampler to be put in Pipeline object in the training part. 
        
            Parameters: 
                y (data series) : Data series of y. 
        
            Returns: 
                Returns the sampler for training part's Pipeline.
        """
        if self.method == "none":
            return None
    
        elif self.method == "oversampling":
            self.sampler = RandomOverSampler()
            return self.sampler

        elif self.method == "undersampling":
            self.sampler = RandomUnderSampler()
            return self.sampler

        elif self.method == "nearmiss":
            self.sampler = NearMiss(version = 1)
            return self.sampler

        elif self.method == "smote":
            self.sampler = SMOTE()
            return self.sampler

        elif self.method == "classweights": #https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/

            if self.class_weights is None:
        
                weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y), y = y)
                self.class_weights = dict(zip(np.unique(y), weights))
            
            
            return self.class_weights
            
        else:
            raise ValueError(f"{self.method} is not a valid method!")

