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

from neuralnetwork import NeuralN

class AutoEncoder(nn.Module):
    """ 
        Class to represent the autoencoder and reflect the customizable pattern. 

        Args: 
            input_dimension (int) : Size of input dimension
            output_dimension (int) : Size of output dimension
            latent_dim (int) : Size of latent dimension.
            hidden_layers (list[int]) : List of hidden layers.
            num_hidden_layers (int) : Amount of hidden layers.
            hidden_dim (int): Default hidden dimension. 
            activation_default (str): Default activation function.
            activations (list[str]) : List of activation functions. 
            loss_method (str) : Loss method to evaluate training and testing. 
            opt_method (str): Optimization method. 
            lr (float): Learning rate. 
            alpha (float): Parameter for focal loss function.
            gamma (float): Parameter for focal loss function.
            epochs (int): Number of epochs. 
            reconstruction_threshold (float): Reconstruction threshold to make predictions.
    """
    def __init__(self, input_dimension = None, output_dimension = None, latent_dim = None, hidden_layers=None, num_hidden_layers = None, hidden_dim = 64,
                 activation_default = "relu",
                 activations = None, loss_method = "BCE", opt_method = "SGD", lr = 0.01, alpha=None, gamma = None, reconstruction_threshold = None, epochs=None):

        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layers = hidden_layers
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.activation_default = activation_default
        self.activations = activations
        self.opt_method = opt_method
        self.lr = lr 
        self.alpha = alpha
        self.gamma = gamma 
        self.reconstruction_threshold=reconstruction_threshold
        self.epochs = epochs
        self.loss_method = loss_method
        self.alpha = alpha
        self.gamma = gamma


        hidden_layers_encoder = None
        if self.hidden_layers is not None:
            
            hidden_layers_encoder = self.hidden_layers

        else:
            hidden_layers_encoder = [self.hidden_dim//(2**i) for i in range(self.num_hidden_layers)]


        hidden_layers_decoder = hidden_layers_encoder[::-1]
            
            
        self.encoder = NeuralN(input_dimension=self.input_dimension, output_dimension=self.latent_dim, hidden_layers = hidden_layers_encoder,  
                               num_hidden_layers=self.num_hidden_layers, hidden_dim=self.hidden_dim, activation_default=self.activation_default,
                               activations = self.activations, loss_method = self.loss_method, opt_method = self.opt_method, lr = self.lr, alpha = self.alpha, gamma = self.gamma, epochs=self.epochs)

        self.decoder = NeuralN(input_dimension=self.latent_dim, output_dimension=self.input_dimension, hidden_layers = hidden_layers_decoder,
                               num_hidden_layers=self.num_hidden_layers, hidden_dim=self.hidden_dim, activation_default=self.activation_default,
                               activations=self.activations, loss_method=self.loss_method, opt_method=self.opt_method, lr=self.lr, alpha=self.alpha, gamma=self.gamma, epochs=self.epochs)


    def forward(self, x):
        """ 
        Forward method to initate the transformation of the input to output.
        
            Parameters: 
                x (tensor) : Training tensor for x. 
        
            Returns: 
                Returns the decoded version of the input.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def get_optimizer(self):
        """ 
            Method to select the optimization method.

        """
        if self.opt_method == "SGD":
            return torch.optim.SGD(params = self.parameters(), lr = self.lr)
    
        elif self.opt_method == "Adam":
            return torch.optim.Adam(params = self.parameters(), lr = self.lr)
    
        elif self.opt_method == "RMSprop":
            return torch.optim.RMSprop(params = self.parameters(), lr = self.lr)
    
        else: 
            raise ValueError(f"{self.opt_method} is not valid!")

    def train_model_ae(self, train_loader, val_loader):
        """ 
            Training phase of the autoencoder.

            Parameters: 
                train_loader (tensor) : Training data loader for training. 
                val_loader (tensor) : Validation data loader for validation.

            Returns: 
                Returns the training and validation loss. 
        """
        print(self.encoder)
        print(self.decoder)
        print("Training starts ! ")
        
        loss_fn = self.encoder.get_loss()
        optimizer = self.get_optimizer()
        size = len(train_loader.dataset)
        t_loss=[]
        val_loss = []
        for e in range(self.epochs):
            self.train()
            train_loss = 0
            for batch, (X, y) in enumerate(train_loader):
       
                output = self(X)
                loss = loss_fn(output, X)
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                train_loss += loss.item() * X.size(0)
            
            train_loss_ = train_loss/len(train_loader.dataset) 
            t_loss.append(train_loss_)
            print(f"Train loss: {t_loss}")
            self.eval()
            test_loss = 0
            with torch.inference_mode():
                for X, y in val_loader: 
                    
                    output=self(X)
                    test_loss += loss_fn(output, X).item()*X.size(0)
                
        
            test_loss_ = test_loss/len(val_loader.dataset)
            val_loss.append(test_loss_)
            print(f"Test loss: {val_loss}")
           
        return t_loss, val_loss
            

    def reconstruction_error(self, test_loader):
        """ 
            A method for calcuating the reconstruction error.

            Parameters: 
                test_loader (tensor) : Test data loader for reconstruction error. 

            Returns: 
                Returns the loss_per_sample and labels. 
        """
        self.eval()
        

        with torch.inference_mode(): 
            loss_per_sample = []
            labels = []
            
           
            for X, y in test_loader:
                output = self(X)
                loss_samp = torch.mean((output - X)**2, dim = 1)

                loss_per_sample.append(loss_samp.numpy())
                labels.append(y.numpy())
            loss_per_sample = np.hstack(loss_per_sample)
            labels = np.hstack(labels)
        
        return loss_per_sample, labels
                
                
    def analysis(self,test_loader):
        """ 
            Analysis of the auto encoder.

            Parameters: 
                test_loader (tensor) : Test data loader for analysis. 

            Returns: 
                Returns a confusion matrix and ROC. 
        """
        error, true_label = self.reconstruction_error(test_loader)

        fpr, tpr, thresholds = metrics.roc_curve(true_label, error)
        auc_ = metrics.auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr)
        plt.plot([0,1], [0,1])
        plt.title("ROC")
        plt.ylabel("TPR")
        plt.xlabel("FPR")
        plt.show()

        predictions = [1 if e > self.reconstruction_threshold else 0 for e in error]
        cm = confusion_matrix(true_label, predictions)

        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual Class")
        plt.xlabel("Predicted Class")

        plt.show()
                      