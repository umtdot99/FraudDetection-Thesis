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

from data_transformation import DataGathering
from trainmodel import TrainModel
from imbalancehandler import ImbalanceHandler
from neuralnetwork import NeuralN
from unsupervisedpart import UnsupervisedTraining
from probnn import PNN
from autoencoder import AutoEncoder
from convolutionalnn import ConvolutionNN
from fraudpipeline import FraudDetectionPipeline
from generator import Generator
from discriminator import Discriminator
from gan_ import Gan



MODEL_DICT = {
    "logistic_regression": LogisticRegression,
    #"random_forest": RandomForestClassifier
    #"svm": SVC,
    #"gaussian_nb": GaussianNB,
    #"decision_trees": DecisionTreeClassifier,
    #"xgb": XGBClassifier,
    #"adaboost": AdaBoostClassifier
}

PARAM_GRID = {
    "logistic_regression": {
        "C": np.logspace(-3, 2, 6),
        "penalty": ["l1", "l2"],
        "solver": ["saga", "liblinear"]  
    },
    "random_forest": {
        "n_estimators": [100],
        "max_depth": [5, 10],
        "min_samples_leaf": [1, 2],
        "criterion": ["gini", "entropy"]
    },
   "svm": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    },
    "gaussian_nb": {
        "var_smoothing": np.logspace(-2, -9, num=3)   
    },
    "decision_trees":{
        "criterion": ["gini", "entropy"]
    },
    "xgb": {
        "n_estimators": [50, 100, 200],
        "max_depth" : [4, 6],
        "learning_rate": [0.01, 0.05, 0.1]
    },


    "adaboost":{
        "learning_rate": [0.1, 0.01]
    }
        
}    

#bin_pipeline here


pipeline_numerical = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy = "median")),
        ('scaler', StandardScaler())
         ]
)


pipeline_categorical = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('one_hot_encoder', OneHotEncoder())
    ]     
)


##%%


data = DataGathering(file_path="/Users/umutkurt/Desktop/Thesis/fraud_detection_project/creditcard.csv", target="Class", batch_size = 64, pipeline_numerical = pipeline_numerical, pipeline_categorical = pipeline_categorical, data_type=torch.float32)
preprocessor = data.get_data_objects()
X_train, y_train, X_test, y_test = data.main_split()
X_train_p = preprocessor.fit_transform(X_train)
cols = preprocessor.get_feature_names_out()

generator = Generator(noise=64, output=30, hidden = 64)
discriminator = Discriminator(input_dim = 30, hidden = 64)
gan = Gan(generator, discriminator, lr_g =0.01, lr_d = 0.01)
imbalance_handler = ImbalanceHandler(method = "gan", gan = gan, gan_epochs=10, gan_noise=64, columns=cols)
aenet = NeuralN(input_dimension = 30, output_dimension=1, hidden_layers=[16,32,64], activations = ["relu", "relu","relu", "relu"], data_type=torch.float32, loss_method="BCE", opt_method = "SGD", lr = 0.01, threshold=0.45, epochs=20)

model = FraudDetectionPipeline(data=data, imbalance_handler = imbalance_handler, neural_networks=aenet)
FraudDetectionPipeline.run_neuralnetwork(model)


#%%


data = DataGathering(file_path="/Users/umutkurt/Desktop/Thesis/fraud_detection_project/creditcard.csv", target="Class", pipeline_numerical = pipeline_numerical, pipeline_categorical = pipeline_categorical, data_type = torch.float32, batch_size = 64)
imbalance_handler = ImbalanceHandler(method = "smote")
neunet = NeuralN(input_dimension = 30, output_dimension = 1, hidden_layers=[64, 128, 64, 32, 16], threshold = 0.3, data_type=torch.float32, activations = ["relu", "tanh","relu", "relu"], loss_method="BCEwLogit", opt_method = "Adam", lr = 0.001, alpha = 0.25, gamma = 2, epochs=10)

model = FraudDetectionPipeline(data=data, imbalance_handler=imbalance_handler, neural_networks=neunet)
FraudDetectionPipeline.run_neuralnetwork(model)


#%%

data = DataGathering(file_path="creditcard.csv", target= "Class", pipeline_numerical = pipeline_numerical, pipeline_categorical = pipeline_categorical, test_size = 0.2, calibration_size = 0.15)
imbalance_handler = ImbalanceHandler(method = "undersampling")
trainer = TrainModel(search_strategy = "grid", param_grid = PARAM_GRID, performance_measure = "average_precision", cv = 4, n_jobs = -1, verbose = 3, dimensionality_reduction= "pca", n_components = 10,  model_dictionary = MODEL_DICT, threshold = 0.45, calibration_needed = "yes")

model = FraudDetectionPipeline(data=data, imbalance_handler=imbalance_handler, trainer=trainer)
FraudDetectionPipeline.run_supervisedmodels(model)


#%%


data = DataGathering(file_path="creditcard.csv", target="Class", pipeline_num_list = ["imputer_num", "scaler"], pipeline_cat_list = ["imputer_cat", "onehot"], data_type = torch.float32, batch_size = 64)
imbalance_handler = ImbalanceHandler(method = "undersampling")
neunet = NeuralN(input_dimension = 30, output_dimension = 1, hidden_layers=[64, 128, 64, 32, 16], threshold = 0.65, data_type=torch.float32, activations = ["relu", "tanh","relu", "relu"], loss_method="MSE", opt_method = "Adam", lr = 0.001, alpha = 0.25, gamma = 2, epochs=10)

model = FraudDetectionPipeline(data=data, imbalance_handler=imbalance_handler, neural_networks=neunet)
FraudDetectionPipeline.run_neuralnetwork(model)


#%%


data = DataGathering(file_path="creditcard.csv", target="Class", pipeline_num_list = ["imputer_num", "scaler"], pipeline_cat_list = ["imputer_cat", "onehot"], data_type = torch.float32, batch_size = 64, val_size = 0.5)
imbalance_handler = ImbalanceHandler(method = "oversampling")
neunet = NeuralN(input_dimension = 30, output_dimension = 1, hidden_layers=[64, 128, 64, 32, 16], threshold = 0.45, data_type=torch.float32, activations = ["relu", "tanh","relu", "relu"], loss_method="focal_loss", opt_method = "Adam", lr = 0.001, alpha = 0.25, gamma = 2, epochs=5)

model = FraudDetectionPipeline(data=data, imbalance_handler=imbalance_handler, neural_networks=neunet)
FraudDetectionPipeline.run_neuralnetwork(model)


#%%


data = DataGathering(file_path="creditcard.csv", target= "Class", pipeline_numerical = pipeline_numerical, pipeline_categorical = pipeline_categorical, test_size = 0.2, calibration_size = 0.15)
imbalance_handler = ImbalanceHandler(method = "classweights", class_weights = {0: 1, 1:10})
trainer = TrainModel(search_strategy = "grid", param_grid = PARAM_GRID, performance_measure = "average_precision", cv = 2, n_jobs = -1, verbose = 3,  model_dictionary = MODEL_DICT, threshold = 0.45, calibration_needed = "yes")

model = FraudDetectionPipeline(data=data, imbalance_handler=imbalance_handler, trainer=trainer)
FraudDetectionPipeline.run_supervisedmodels(model)


#%%


data = DataGathering(file_path="creditcard.csv", target="Class", batch_size = 64, pipeline_numerical = pipeline_numerical, pipeline_categorical = pipeline_categorical, data_type = torch.float32)
imbalance_handler = ImbalanceHandler(method = "undersampling")
cnn = ConvolutionNN(layer_list   = [1, 16, 16, 32],pool  = [2,  2],activations  = ['relu', 'relu'], kernel= 3, padding = 1,dropout_list = [0.3, 0.3], flattened_size = 96,linear_layer = 64, output_dimension = 1, loss_method = "BCEwLogit", threshold=0.65,  opt_method = "Adam", lr = 0.01)

model = FraudDetectionPipeline(data = data, imbalance_handler = imbalance_handler, convolutional_nn = cnn)
FraudDetectionPipeline.run_cnn(model)


#%%

data = DataGathering(file_path="creditcard.csv", target="Class", pipeline_numerical = pipeline_numerical,pipeline_categorical = pipeline_categorical)
imbalance_handler = ImbalanceHandler(method = "undersampling")

model = FraudDetectionPipeline(data = data, imbalance_handler = imbalance_handler)
FraudDetectionPipeline.run_pnn(model, kernel="gaussian", bandwidth=0.2)


#%%


data = DataGathering(file_path="creditcard.csv", target="Class", batch_size = 64, pipeline_numerical = pipeline_numerical, pipeline_categorical = pipeline_categorical, autoencoder = "yes")
imbalance_handler = ImbalanceHandler(method = "none")
aenet = AutoEncoder(input_dimension = 30, latent_dim = 4, hidden_layers=[16,8,4], hidden_dim=16, activations = ["relu", "relu","relu", "relu"], loss_method="MSE", opt_method = "SGD", lr = 0.01, reconstruction_threshold=1.9, epochs=10)

model = FraudDetectionPipeline(data=data, imbalance_handler = imbalance_handler, auto_encoders=aenet)
FraudDetectionPipeline.run_autoencoder(model)


#%%

DISTANCE_METRICS = {
        "euclidean" : type_metric.EUCLIDEAN,
        "squared euclidean" : type_metric.EUCLIDEAN_SQUARE,
        "manhattan" : type_metric.MANHATTAN,
        "chebyshev": type_metric.CHEBYSHEV,
        "canberra" : type_metric.CANBERRA,
        "chi_square" : type_metric.CHI_SQUARE
        
}

METHODS = ['kmeans', 'dbscan']
    
    #Unsupervised learning
data = DataGathering(file_path="trainfinal.csv", pipeline_numerical = pipeline_numerical, pipeline_categorical = pipeline_categorical)
trainer_uns = UnsupervisedTraining(n_clusters = 3, 
                                    eps = 3, min_samples = 3)
    
model = FraudDetectionPipeline(data=data, unsupervised_clustering=trainer_uns)
FraudDetectionPipeline.run_unsupervisedclustering(model)

