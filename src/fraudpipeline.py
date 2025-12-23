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

import torch
import numpy as np
import pandas as pd 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns

class FraudDetectionPipeline(nn.Module):
    """ 
        Class to represent the pipeline for fraud detection. 

        Args: 
            data (data object) : DataGathering Class Object.
            trainer (trainer object) : Trainer Object.
            neural_networks (neural network object) : Neural Network Object.
            imbalance_handler (imbalance handler object) : List of hidden layers.
            convolutional_nn (cnn object) : CNN object.
            auto_encoders (ae object): Autoencoder object.
            unsupervised_clustering (unsupervised object): Unsupervised training object.
            probabilistic_neural_network (pnn object) : Probabilistic Neural Network Object.
        
    """
    def __init__(self, data=None, trainer=None, neural_networks=None, imbalance_handler=None, convolutional_nn=None, auto_encoders=None, unsupervised_clustering=None, probabilistic_neural_network=None):

        super().__init__()
        self.data = data
        self.trainer = trainer
        self.neural_networks = neural_networks 
        self.auto_encoders = auto_encoders
        self.unsupervised_clustering = unsupervised_clustering
        self.probabilistic_neural_network = probabilistic_neural_network
        self.imbalance_handler = imbalance_handler
        self.convolutional_nn = convolutional_nn


    def run_neuralnetwork(self):
        """ 
        Running of the neural networks.

        """
        train_loader, val_loader, test_loader = self.data.data_loading(imbalance_handler = self.imbalance_handler)
        t, v = self.neural_networks.train_model(train_loader, val_loader)
        test = self.neural_networks.test_model(val_loader)
        self.neural_networks.analysis(t, v)
        prob, preds, labels = self.neural_networks.predict(test_loader)
        self.neural_networks.report(test_loader, preds, labels)

        print("Finished")

    def run_supervisedmodels(self):
        """ 
        Running of the supervised models.

        """
        X_train, y_train, X_test, y_test = self.data.main_split()
        preprocessor_ = self.data.get_data_objects()
        X_train_, y_train_, X_calib, y_calib = self.data.calibration_loading(X_train, y_train)
        self.trainer.fitting_every_model(X_train, y_train, preprocessor_, self.imbalance_handler)
        self.trainer.analysis(X_test, y_test, X_train, y_train, X_calib, y_calib) 

        print("Finished")

    def run_cnn(self):
        """ 
        Running of the CNN.

        """
        train_loader, val_loader, test_loader = self.data.data_loading(imbalance_handler = self.imbalance_handler)
        t, v = self.convolutional_nn.train_model(train_loader, val_loader)
        self.convolutional_nn.analysis(t, v)
        prob, preds, labels = self.convolutional_nn.predict(test_loader)
        self.convolutional_nn.report(test_loader, preds, labels)

        print("Finished")

    def run_pnn(self, kernel=None, bandwidth=None):
        """ 
        Running of the PNN.

        """
        X_train_, y_train, X_test_, y_test = self.data.main_split()
        preprocessor_ = self.data.get_data_objects()
        X_train_df = preprocessor_.fit_transform(X_train_)
        X_test_df = preprocessor_.transform(X_test_)
        pnn = PNN(X_train_df, y_train, imbalance_handler, kernel = kernel, bandwidth = bandwidth)
        predictions = pnn.pnn_pred(X_test_df)
        f1 = f1_score(y_test.values, predictions, average="weighted")

        pnn.analysis(y_test.values, X_test_df)
        
        print("Finished")

    def run_autoencoder(self):
        """ 
        Running of the Autoencoders.

        """
        train_loader, val_loader, test_loader = self.data.data_loading(imbalance_handler=self.imbalance_handler)
        self.auto_encoders.train_model_ae(train_loader, val_loader)
        self.auto_encoders.analysis(test_loader)
        
        print("Finished")

    def run_unsupervisedclustering(self):
        """ 
        Running of the unsupervised part.

        """
        X_train, X_test = self.data.main_split()
        preprocessor_ = self.data.get_data_objects()
        best_model = self.unsupervised_clustering.clustering_results(X_train, X_test, preprocessor_)
        print(best_model)

        print("Finished")
