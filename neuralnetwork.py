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

from focalloss import FocalLoss
from mfeloss import MFELoss
from msfeloss import MSFELoss


class NeuralN(nn.Module):
    """ 
        Class to represent the autoencoder and reflect the customizable pattern. 

        Args: 
            input_dimension (int) : Size of input dimension
            output_dimension (int) : Size of output dimension
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
            threshold (float): Threshold to make predictions.
            class_weights (dict): Class weights for imbalanced sets.
            data_type (pytorch object): Data type to avoid confusion.
    """
    def __init__(self, input_dimension = None, output_dimension = None, hidden_layers=None, num_hidden_layers = None, hidden_dim = 64,
                 activation_default = "relu", threshold = 0.3,
                 activations = None, loss_method = "BCE", opt_method = "SGD", lr = 0.01, class_weights = None, alpha=None, data_type = None, gamma = None, epochs=None):
        
        super().__init__() 
        
        self.loss_method = loss_method
        self.opt_method = opt_method
        self.lr = lr 
        self.alpha = alpha 
        self.gamma = gamma 
        self.epochs = epochs
        self.input_dimension = input_dimension 
        self.output_dimension = output_dimension
        self.hidden_layers = hidden_layers #list
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim #default
        self.activation_default = activation_default #default 
        self.activations = activations #list
        self.data_type = data_type
        self.threshold = threshold
        
        if class_weights is not None:
            self.class_weights = torch.tensor([class_weights[1], class_weights[0]], dtype= torch.float32)
        else: 
            self.class_weights = class_weights
        
        self.process = nn.ModuleList()

        layer = None
        if self.hidden_layers is not None:
            
            layer = [self.input_dimension] + self.hidden_layers + [self.output_dimension]

        else: 

            layer = [self.input_dimension] + [self.hidden_dim]*self.num_hidden_layers + [self.output_dimension]

        act = None
        if self.activations is not None:
            
            if len(self.activations) < (len(layer)):
                need = (len(layer)) - len(self.activations)
                act = self.activations + ["identity"]*need 
            
            else: 
                act = self.activations

        else: 

            act = [self.activation_default]* (len(layer) - 1) 

        for i in range(1, len(layer)):
            self.process.append(nn.Linear(layer[i-1], layer[i]))
            
            if i < (len(layer) - 1):
                self.process.append(self.get_activation(act[i-1]))
            
            elif i == (len(layer) - 1):

                if self.loss_method == "BCE": 
                    self.process.append(nn.Sigmoid())

                else: 
                    self.process.append(nn.Identity())

    
    def forward(self, x):
        """ 
        Forward method to initate the transformation of the input to output.
        
            Parameters: 
                x (tensor) : Training tensor for x. 
        
            Returns: 
                Returns the processed version of the input.
        """
        x = x.float()
        for m in self.process:
            x = m(x)
        return x

    def get_activation(self, activation_):
        """ 
            Method to select the activation function.

            Parameters: 
                activation_ (str): Name of the activation function as a string.

            Returns: 
                Returns the activation function with nn module. 

        """
        if activation_ == "relu": 
            return nn.ReLU()

        elif activation_ == "tanh": 
            return nn.Tanh()

        elif activation_ == "identity": 
            return nn.Identity()
        
    def get_loss(self):
        
        """ 
            Method to select the loss function.

        """
        
        if self.loss_method == "BCE":
            return nn.BCELoss()
        
        elif self.loss_method == "L1":
            return nn.L1Loss()
        
        elif self.loss_method == "MSE":
            return nn.MSELoss()

        elif self.loss_method == "CE":
            return nn.CrossEntropyLoss(weight = self.class_weights)

        elif self.loss_method == "BCEwLogit":
            if self.class_weights is not None:
                pos_weight = torch.tensor([self.class_weights[1] / self.class_weights[0]])
                return nn.BCEWithLogitsLoss(pos_weight = pos_weight)
            else: 
                return nn.BCEWithLogitsLoss()

        elif self.loss_method == "focal_loss": 
            return FocalLoss(alpha=self.alpha, gamma=self.gamma)

        elif self.loss_method == "MFE":
            return MFELoss()

        elif self.loss_method == "MSFE":
            return MSFELoss()

        else:
            raise ValueError(f"{self.loss_method} is not valid!")
    
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
    
    def train_model(self, train_loader, val_loader):
        """ 
            Training phase of the NN.

            Parameters: 
                train_loader (tensor) : Training data loader for training. 
                val_loader (tensor) : Validation data loader for validation.

            Returns: 
                Returns the training and validation loss. 
        """
        loss_fn = self.get_loss()
        optimizer = self.get_optimizer()        
        size = len(train_loader.dataset)
        t_loss=[]
        val_loss = []
        for e in range(self.epochs):
            self.train()
            train_loss = 0
            for batch, (X, y) in enumerate(train_loader):
                
                y_logits = self(X).squeeze()
                
                loss = loss_fn(y_logits, y)
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                train_loss += loss.item() * X.size(0)
            train_loss_ = train_loss/len(train_loader.dataset) 
            t_loss.append(train_loss_)
            
            self.eval()
            test_loss = 0
            with torch.inference_mode():
                for X, y in val_loader: 
                    
                    y_logits = self(X).squeeze()
                    test_loss += loss_fn(y_logits, y).item() * y.size(0)
                
        
            test_loss_ = test_loss/len(val_loader.dataset)
            val_loss.append(test_loss_)
    
        return t_loss, val_loss
        
            
    def test_model(self, test_loader, loss_fn=None):
        """ 
            A method for calcuating the test error.

            Parameters: 
                test_loader (tensor) : Test data loader for reconstruction error. 

            Returns: 
                Returns the loss_per_sample and labels. 
        """
        loss_fn = self.get_loss()
        test_loss = []

        for e in range(self.epochs): 
            self.eval()
            with torch.inference_mode():
 
                
                test_loss_num = 0
                for X, y in test_loader: 
                    y_logits = self(X).squeeze()
                    test_loss_num += loss_fn(y_logits, y).item() * y.size(0)
                    
                test_loss_ = test_loss_num/len(test_loader.dataset)
                test_loss.append(test_loss_)
        return test_loss

    def store(self, operation=None, path = None): #https://pytorch.org/tutorials/beginner/saving_loading_models.html

        """ 
            A method for storing or loading the model.

            Parameters: 
                operation (str) : Name of the operation. 
                path (str) : Name of the path.
 
        """
        if operation == "save": 
            torch.save(self.state_dict(), path)
    
        elif operation == "load": 
            self.load_state_dict(torch.load(path, weights_only = True))
            print("Loading successfull ! ")

    def analysis(self, l1, l2):
        """ 
            Analysis of the Neural Network.

            Parameters: 
                l1 (list) : List of loss in training.
                l2 (list) : List of loss in validation.

            Returns: 
                Returns graph that includes validation and training error. 
        """
        l1_arr = np.array(l1)
        l2_arr = np.array(l2)

        epoch_l = [i for i in range(1, self.epochs + 1)]
        epoch_arr = np.array(epoch_l)

        sns.lineplot(x = epoch_arr, y = l1_arr, label="Train Error")
        sns.lineplot(x = epoch_arr, y = l2_arr, label="Valid Error")
        plt.legend()
        plt.show()


    def predict(self, test):
        """ 
            Prediction with NN.

            Parameters: 
                test (tensor): Test loader for prediction. 
                

            Returns: 
                Returns probabilities, predictions and labels. 
        """
        predictions=[]
        labels = []
        probs=[]
        if self.loss_method == "BCE": 
            
            self.eval()
            with torch.inference_mode():
                for X, y in test:
                    
                    output = self(X).squeeze()
                    preds = (output > self.threshold).int()
                    predictions.append(preds)
                    labels.append(y)
                    probs.append(output)
                    
        elif self.loss_method == "CE": 
            self.eval()
            with torch.inference_mode():
                for X, y in test:
                    output = self(X).squeeze()
                    prob = torch.argmax(output, dim=1)
                    preds = (prob>self.threshold).int()
                    predictions.append(preds)
                    labels.append(y)
                    probs.append(prob)
                    
        else:
            self.eval()
            with torch.inference_mode():
                for X, y in test:
                    output = self(X).squeeze()
                    prob = torch.sigmoid(output)
                    preds = (prob > self.threshold).int()
                    predictions.append(preds)
                    labels.append(y)
                    probs.append(prob)

        all_preds = torch.cat(predictions).ravel()
        all_labels = torch.cat(labels).ravel()
        all_probs = torch.cat(probs).ravel()
        
        return all_probs, all_preds, all_labels
        
    def report(self, test, pred, labels): 
        """ 
            Reporting part for NN.

            Parameters: 
                test (tensor): Test loader for prediction.
                pred (tensor): Tensor for predictions.
                labels (tensor): True labels.
                

            Returns: 
                Returns a confusion matrix. 
        """
        all_probs, all_preds, all_labels = self.predict(test)
        all_probs_arr = all_probs.detach().numpy().ravel()
        all_preds_arr = all_preds.detach().numpy().ravel()
        all_labels_arr = all_labels.detach().numpy().ravel()
        
        cm = confusion_matrix(all_labels_arr, all_preds_arr)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual Class")
        plt.xlabel("Predicted Class")

        plt.show()
            