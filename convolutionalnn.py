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



class ConvolutionNN(nn.Module):
    """ 
        Class to initiate the Convolutional Neural Network object. 

        Args: 
            input_dimension (int) : Size of input dimension
            output_dimension (int) : Size of output dimension
            kernel (int) : Size of the kernel.
            padding (int) : Size of the padding.
            dropout_rate (float): Size of the drop out rate.
            layer_list (list[int]): List that includes the layers of the convolutional network.
            dropout_list (list[float]): List of drop out rates. 
            pool (list[int]): List of pool with numbers. 
            layer_count (int): Desired number of layers in the structure.
            activations (list[str]) : List of activation functions.
            default_layer (int): Dimension of the layer default. 
            linear_layer (int): Dimension of the linear layer.
            flattened_size (int): Size of the flattening.
            loss_method (str) : Loss method to evaluate training and testing. 
            opt_method (str): Optimization method. 
            lr (float): Learning rate. 
            class_weights (dict[int]: float): Class weights for imbalance handling. 
            epochs (int): Number of epochs. 
            threshold (float): Threshold level for predictions
            
    """
    def __init__(self, loss_method = "BCE", opt_method = "SGD", lr = 0.01,
                 threshold = 0.3, class_weights = None, epochs=5, input_dimension = None, dropout_list=None, linear_act = None, flattened_size = None, linear_layer = None, 
                 layer_list = None,default_layer=None, pool=None, activations=None, layer_count = None, dropout_rate= None, output_dimension= None, kernel = 3, padding=1):
        super().__init__()
        
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.kernel = kernel
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.layer_list = layer_list
        self.dropout_list = dropout_list
        self.pool = pool
        self.layer_count = layer_count
        self.activations = activations
        self.default_layer = default_layer
        self.linear_layer = linear_layer
        self.flattened_size = flattened_size
        self.loss_method = loss_method
        self.opt_method = opt_method
        self.lr = lr 
        self.class_weights = class_weights
        self.epochs = epochs
        self.threshold = threshold
        
        if self.class_weights is not None:
            self.class_weights = torch.tensor([class_weights[1], class_weights[0]], dtype= torch.float32)
        else: 
            self.class_weights = class_weights
            
        layer_dict = {}
        
        if self.layer_list is None: 
            
            layer = [self.input_dimension] + [self.default_layer]*self.layer_count + [self.output_dimension]

        else: 
            
            layer = self.layer_list

        for i in range(1, len(layer)):

            layer_dict[f"conv{i}"] = nn.Conv1d(layer[i-1], layer[i], kernel_size = self.kernel, padding = self.padding)

            act = None
            if self.activations is None:
                
                layer_dict[f"act{i-1}"] = nn.ReLU()

            else: 

                if len(self.activations) < (len(layer)):
                    need_act = (len(layer)) - len(self.activations)
                    act = self.activations + ["identity"]*need_act 
                    
                layer_dict[f"act{i-1}"] = self.get_activation(act[i-1])

            pooling = None            
            if self.pool is None:
                
                layer_dict[f"pooling{i-1}"] = nn.MaxPool1d(2,2)

            else: 
                
                if len(self.pool) < (len(layer)):
                    need_pool = (len(layer)) - len(self.pool)
                    pooling = self.pool + [2]*need_pool
                layer_dict[f"pooling{i-1}"] = nn.MaxPool1d(pooling[i-1], pooling[i])

            drop = None
            if self.dropout_list is None: 
                
                layer_dict[f"dropout{i-1}"] = nn.Dropout(self.dropout_rate)

            else: 
                if len(self.dropout_list) < (len(layer)):
                    need_drop = (len(layer)) - len(self.dropout_list)
                    drop = self.dropout_list + [0.3]*need_drop 
                layer_dict[f"dropout{i-1}"] = nn.Dropout(drop[i-1])

    
        
        self.linear1 = nn.Linear(self.flattened_size, self.linear_layer)
        self.linear2 = nn.Linear(self.linear_layer, self.output_dimension)

        self.process = nn.ModuleDict(layer_dict)
        print(self.process)
    

    def forward(self, x):
        """ 
        Forward method to initate the transformation of the input to output.
        
            Parameters: 
                x (tensor) : Training tensor for x. 
        
            Returns: 
                Returns the processed version of the input.
        """
        for m in self.process.values():
            x = m(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        print(x)
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
            Training phase of the CNN.

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
                X = X.unsqueeze(1)
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
                    
                    X = X.unsqueeze(1)
                    y_logits = self(X).squeeze()
                    test_loss += loss_fn(y_logits, y).item() * y.size(0)
                
        
            test_loss_ = test_loss/len(val_loader.dataset)
            val_loss.append(test_loss_)
            print(f"Test loss: {val_loss}")
           
        return t_loss, val_loss

    def analysis(self, l1, l2):
        """ 
            Analysis of the auto encoder.

            Parameters: 
                l1 (list) : List of training error.
                l2 (list) : List of validation error.
                

            Returns: 
                Returns a graph that includes validation and training error in epochs. 
        """
        l1_arr = np.array(l1)
        l2_arr = np.array(l2)

        epoch_l = [i for i in range(1, self.epochs + 1)]
        epoch_arr = np.array(epoch_l)

        sns.lineplot(x = epoch_arr, y = np.log(l1_arr), label="Train Error")
        sns.lineplot(x = epoch_arr, y = np.log(l2_arr), label="Valid Error")
        plt.legend()
        plt.show()


    def predict(self, test):
        """ 
            Prediction with CNN.

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
                    X = X.unsqueeze(1)
                    output = self(X).squeeze()
                    preds = (output > self.threshold).int()
                    predictions.append(preds)
                    labels.append(y)
                    probs.append(output)
                    
        elif self.loss_method == "CE": 
            self.eval()
            with torch.inference_mode():
                for X, y in test:
                    X = X.unsqueeze(1)
                    output = self(X)
                    prob = torch.argmax(output, dim=1)
                    preds = (prob>self.threshold).int()
                    predictions.append(preds)
                    labels.append(y)
                    probs.append(prob)
                    
        else:
            self.eval()
            with torch.inference_mode():
                for X, y in test:
                    X = X.unsqueeze(1)
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
            Reporting part for CNN.

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