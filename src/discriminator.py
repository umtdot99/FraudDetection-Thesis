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

class Discriminator(nn.Module):

    """
        Class to represent the discriminator for Generative Adversarial Networks (GANs).

        Attributes: 
            input_dim (int): Input dimension for the neural network layers.
            hidden (int): Hidden units for hidden layers. 
    """
    
    def __init__(self, input_dim = 30, hidden = 64):
    
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        
        self.discriminator = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden*4),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Linear(self.hidden*4, self.hidden*2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(self.hidden*2, self.hidden),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(self.hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
    
        """
            Function to apply desired layers and activation functions. 
    
            Parameters: 
                x (torch.tensor) : Input tensor. 
                
            Returns: 
                torch.tensor : Logits or probabilities. 
        """

        
        return self.discriminator(x)

