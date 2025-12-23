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

class MSFELoss(nn.Module):

    """
        Class to represent the Mean Squared False Error loss. 

    """
    
    def __init__(self):
        
        super(MSFELoss, self).__init__()

    def forward(self, pred, target):

        """
            Forward function to create the loss function to be used in neural networks. 

            Args:
                pred (torch.tensor): Tensor for predicted values.
                target (torch.tensor): Tensor for target values.

            Returns: 
                loss: The Mean Squared False Error loss for imbalanced datasets. 
        """
        
        yi = torch.sigmoid(pred)

        error = ((target - yi)**2)/2

        N_tot = torch.sum(error*(target==0), dtype=torch.float32)
        P_tot = torch.sum(error*(target==1), dtype=torch.float32)

        fne = N_tot / torch.sum((target==0), dtype=torch.float32)
        fpe = P_tot / torch.sum((target==1), dtype=torch.float32)

        loss = fne**2 + fpe**2
        
        return loss

