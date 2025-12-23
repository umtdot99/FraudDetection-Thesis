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

class Gan(nn.Module):

    """
        Class to represent the Generative Adversarial Network (GAN) structure. 

        Attributes: 
            generator (torch.tensor): Tensor for generator.
            discriminator (torch.tensor): Tensor for discriminator.
            lr_g (float): Learning rate for the generator.
            lr_d (float): Learning rate for the discriminator. 
            optimize_generator (torch object): Optimization type for generator.
            optimize_discriminator (torch object): Optimization type for discriminator.
            criterion_d (nn object): Loss function for discriminator.
            criterion_g (nn object): Loss function for generator.
            noise (int): Noise for generator. 
            epochs (int): Number of epochs. 
    """
    
    def __init__(self, generator, discriminator, lr_g = 0.01, lr_d = 0.01, noise=64, epochs=5):

        super(Gan, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.noise = noise
        self.epochs=epochs

        self.lr_g = lr_g
        self.lr_d = lr_d

        self.optimize_generator = torch.optim.SGD(self.generator.parameters(), lr = self.lr_g)
        self.optimize_discriminator = torch.optim.SGD(self.discriminator.parameters(), lr = self.lr_d)

        self.criterion_d = nn.BCELoss()
        self.criterion_g = nn.BCELoss()

    def train(self, train_data_loader):

        """
            Trains the neural network with given data loader.

            Parameters: 
                train_data_loader (torch.tensor): Data loader for the training phase.

            Returns: 
                Returns the model.
        """

        for epoch in range(self.epochs): 
            for i, (X, y) in enumerate(train_data_loader):


                for param in self.generator.parameters():
                    param.requires_grad = False

                for param in self.discriminator.parameters():
                    param.requires_grad = True


                self.optimize_discriminator.zero_grad()

                batch_size = X.size(0)
                noise_dim = torch.randn(batch_size, self.noise)
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                discriminator_real = self.discriminator(X)
                loss_disc = self.criterion_d(discriminator_real, real_labels)

                fake_ = self.generator(noise_dim)
                disc_fake = self.discriminator(fake_.detach()) 

                loss_fake_disc = self.criterion_d(disc_fake, fake_labels)

                total_disc_loss = (loss_disc + loss_fake_disc)/2

                total_disc_loss.backward()
                self.optimize_discriminator.step()


                for param in self.generator.parameters():
                    param.requires_grad = True

                for param in self.discriminator.parameters():
                    param.requires_grad = False


                self.optimize_generator.zero_grad()

                noise_dim = torch.randn(batch_size, self.noise)

                fake_data = self.generator(noise_dim)

                labels_for_gen = torch.ones(batch_size, 1)

                disc_pred = self.discriminator(fake_data)

                loss_gen = self.criterion_g(disc_pred, labels_for_gen)

                loss_gen.backward()
                self.optimize_generator.step()

        return self.generator

    def creating_new(self, samples_needed):

        """
            Creates the new samples with using the trained generator. 

            Args: 
                samples_needed (int): Needed samples to solve the issue of class imbalances.  

            Returns: 
                new_samples (torch.tensor): Returns new samples as a tensor. 
        """
        self.generator.eval()
        with torch.no_grad():
            
            noise_generate = torch.randn(samples_needed, self.noise)
            new_samples = self.generator(noise_generate).detach().numpy()

        return new_samples
    
        