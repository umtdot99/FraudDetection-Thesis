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

class UnsupervisedTraining:
    """ 
            Initializes the unsupervised training object. 

            Parameters: 
                n_cluster (int) : Number of clusters used by pyclustering. 
                eps (int) : DBSCAN eps parameter. 
                min_samples (int) : Number of min samples. 
                
        """
    def __init__(self, n_clusters = 3, eps = 3, min_samples = 3):

        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples

    def preprocessing_unsupervised(self, X_train, X_test, preprocessor): 

        """ 
            Fits every model stated in the model dictionary.

            Parameters: 
                X_train (data frame) : Training data for X. 
                X_test (data frame) : Test data for y. 
                preprocessor (object): Preprocesser that comes from data gathering module. 
            
            Returns: 
                Returns the X_train and X_test with their pre-processed forms. 
        """
        X_train_prep = preprocessor.fit_transform(X_train)
        X_test_prep = preprocessor.transform(X_test)

        return X_train_prep, X_test_prep
        
    def clustering_visualization(self, X):
        """ 
            Fits every model stated in the model dictionary.

            Parameters: 
                
                X (data frame) : Training data for X. 
                
            Returns: 
                Returns the graph for the visualization of the clusters.  
        """
        X_arr = self.X.to_numpy()
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = 45, n_iter = 400)
        X_2d = tsne.fit_transform(X_arr)
        X_list = X_2d.tolist()

        results = []
        for method in METHODS: 
            for metric in DISTANCE_METRICS:

                if method == "kmeans": 
                    
                    initials = random_center_initializer(X_list, self.n_clusters).initialize()
        
                    k_means = kmeans(X_list, initials, ccore=False, metric = distance_metric(DISTANCE_METRICS[metric]))
                    k_means.process()
                    clusters = k_means.get_clusters()
                    centers = k_means.get_centers()
                    kmeans_visualizer.show_clusters(X_list, clusters, centers)
                    score_k_mean = silhouette(X_list, clusters).process().get_score()
                    score_k_mean_avg = np.mean(score_k_mean)

                    results.append({"method": method, "metric": metric, "silhouette_avg": score_k_mean_avg})

                if method == "dbscan":
                    db_scan = dbscan(X_list, self.eps, self.min_samples, ccore=False, metric = distance_metric(DISTANCE_METRICS[metric]))
                    db_scan.process()

                    clusters = db_scan.get_clusters()
                    noise = db_scan.get_noise()
                    score_db = silhouette(X_list, clusters).process().get_score()
                    score_db_avg = np.mean(score_db)
                    visualization = cluster_visualizer()
                    visualization.append_clusters(clusters, X_list)
                    visualization.append_clusters(noise, X_list, marker='x')
                    visualization.show()
                    
     

    def clustering_results(self, X_train, X_test, preprocessor):
        """ 
            Fits every model stated in the model dictionary.

            Parameters: 
                
                X_train (data frame) : Training data for X.
                X_test (data frame) : Testing data for X.
                preprocessor (object): Preprocesser that comes from data gathering module. 
                
            Returns: 
                Returns the best model with silhouette score.  
        """
        X_list_, X_test_ = self.preprocessing_unsupervised(X_train, X_test, preprocessor)
        X_list = X_list_.tolist()
        X_test = X_test_.tolist()
        results_view = []
        results = {}
        minimum = -9999
        for method in METHODS: 
            for metric in DISTANCE_METRICS: 
                if method == "kmeans": 
                    initials = random_center_initializer(X_list, self.n_clusters).initialize()
        
                    k_means = kmeans(X_list, initials, ccore=False, metric = distance_metric(DISTANCE_METRICS[metric]))
                    k_means.process()
                    clusters = k_means.get_clusters()
                    centers = k_means.get_centers()
                    score_k_mean = silhouette(X_list, clusters).process().get_score()
                    score_k_mean_avg = np.mean(score_k_mean)
                    results_view.append({"method": method, "metric": metric, "silhouette_avg": score_k_mean_avg})
                    if score_k_mean_avg >= minimum: 
                        minimum = score_k_mean_avg
        
                if method == "dbscan": 

                    db_scan = dbscan(X_list, self.eps, self.min_samples, ccore=False, metric = distance_metric(DISTANCE_METRICS[metric]))
                    db_scan.process()

                    clusters = db_scan.get_clusters()
                    noise = db_scan.get_noise()
                    score_db = silhouette(X_list, clusters).process().get_score()
                    score_db_avg = np.mean(score_db)
                    results_view.append({"method": method, "metric": metric, "silhouette_avg": score_db_avg})
                    if score_db_avg >= minimum: 
                        minimum = score_db_avg
        print(results_view)
        for i in results_view: 
            if i["silhouette_avg"] == minimum: 
                best_model = i
        
        return best_model


# ## References

# 1- Chillar, A. *K-means clustering using different metrics*. Accessed on March 13, 2025, from https://www.kaggle.com/code/arushchillar/kmeans-clustering-using-different-distance-metrics
# 
# 2- PyClustering Development Team. *PyClustering 0.8.2 documentation: Examples. Accessed on March 13, 2025, from https://pyclustering.github.io/docs/0.8.2/html/index.html#example_sec
