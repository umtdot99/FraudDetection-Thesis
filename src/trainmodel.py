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

class TrainModel:
    """ 
            Initializes the training object. 

            Parameters: 
                param_grid (dict) : Dictionary of the parameter grid supplied by the user 
                search_strategy (str) : Hyperparameter tuning strategy to optimize the given parameter grid. 
                model_dictionary (dict) : Dictionary of models. 
                performance_measure (str): Performance to be considered in the hyperparameter tuning.
                cv (int) : Amount of cross validation.
                n_jobs (int) : Concurrent processes that are being used. 
                verbose (int) : Extra information during the process.
                threshold (float) : Threshold for prediction.
                calibration_needed (str): If yes, calibration will be applied. 
                interaction (str): Interaction term for logistic regression.
                dimensionality_reduction (str): Type of dimensionality reduction. 
                n_components(int): Number of components in dimensionality reduction technique.
                
        """
    def __init__(self, param_grid = None, search_strategy = "grid", model_dictionary = None,
                performance_measure = "f1", cv = 3, n_jobs = -1, verbose = 1, 
                 threshold = None, calibration_needed=None, interaction=None, dimensionality_reduction=None, n_components = None):
        
    
        self.search_strategy = search_strategy
        self.performance_measure = performance_measure
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.param_grid = param_grid

        self.model_dictionary = model_dictionary
        self.threshold = threshold
        self.calibration_needed=calibration_needed
        self.interaction = interaction
        self.dimensionality_reduction = dimensionality_reduction
        self.n_components = n_components
        self.best_estimator_b = None
        self.best_score_b = None
        self.best_model = None
        
        

    def fitting_every_model(self, X, y, preprocessor, imbalance_handler = "none"):
        
        """ 
            Fits every model stated in the model dictionary.

            Parameters: 
                X (data frame) : Training data for X. 
                y (data series) : Training data for y. 
                imbalance_handler (str) : Imbalance handler technique that is taken from imbalancehandler class. 
                preprocessor (object): Preprocesser that comes from data gathering module. 
            Returns: 
                Returns best estimator. 
        """
        scores = {}
        performance_list = {}
        best_score_o = -1000
        best_estimator = None
        best_model = None
     
        sampler_ = imbalance_handler.sampler_(y)
        
        poly = PolynomialFeatures(degree=2,
                          interaction_only=True)

        for name, models in self.model_dictionary.items():

            starting_model = models()
            grid = self.param_grid[name].copy() 
            steps= []
            
            if preprocessor is not None: 
                
                steps.append(("preprocessor", preprocessor))
            
            if (name=="logistic_regression") and (self.interaction is not None):
                 
                steps.append(("poly", poly))

            if self.dimensionality_reduction == "pca": 
                steps.append(("pca", PCA(n_components = self.n_components)))
            elif self.dimensionality_reduction == "umap": 
                steps.append(("tsne", umap.UMAP(n_neighbors = self.n_components)))

            else: 
                pass
            
            if sampler_ is not None:
                
                if imbalance_handler.method == "classweights": 
                    steps.append((name, starting_model))
                    model = Pipeline(steps=steps)
                    grid = {f"{name}__{param_name}": param_value for param_name, param_value in grid.items()}
                    grid[f"{name}__class_weight"]= [sampler_]

                else: 
                    steps.append(("sampler", sampler_))
                    steps.append((name, starting_model))
                    
                    model = Pipeline(
                       steps=steps)

                    grid = {f"{name}__{param_name}": param_value for param_name, param_value in grid.items()}
                    
            
            else:
                steps.append((name, starting_model))
                model = Pipeline(steps=steps)
                grid = {f"{name}__{param_name}": param_value for param_name, param_value in grid.items()}
                
            if name == "adaboost": 
                print("I am using AdaBoost !")
                if isinstance(model, Pipeline):  
                    grid[f"{name}__estimator"] = [SVC(C = 1, kernel="rbf"),
                                                 LogisticRegression(penalty="l2"),
                                                 RandomForestClassifier(n_estimators = 300)]
                else:
                    grid["estimator"] = [SVC(C = 1, kernel="rbf"),
                                         LogisticRegression(penalty="l2"),
                                         RandomForestClassifier(n_estimators = 300)]      
            
            
            if self.search_strategy == "grid":
                print(model, grid)
                opt = GridSearchCV(model, grid, scoring=self.performance_measure, 
                               cv = self.cv, n_jobs = self.n_jobs, verbose = self.verbose)
    
            elif self.search_strategy == "random": 
                print(model, grid)
                opt = RandomizedSearchCV(model, grid, scoring = self.performance_measure,
                                    cv = self.cv, n_jobs = self.n_jobs, verbose = self.verbose)
            
            opt.fit(X, y)
            performance_list[name] = opt
            scores[name] = opt.best_score_
            
        print("*****************************************")
        print("Best scores for each model")
        print(performance_list)
        print(f"Scores **************** {scores}")
    
        best_model = max(performance_list, key= lambda k: performance_list[k].best_score_) 
        best_score_o = performance_list[best_model].best_score_
        best_estimator = performance_list[best_model].best_estimator_
        
        self.best_estimator_b = best_estimator
        self.best_score_b = best_score_o
        self.best_model = best_model
        
        print(f"Best model: {best_model} with respect to {self.performance_measure} of {performance_list[best_model]}")
        return self.best_estimator_b
            

    def calibration(self, X_train, y_train, X_calib, y_calib):
        """ 
            Calibration of the probabilities.

            Parameters: 
                X_train (data frame) : Training data for X. 
                y_train (data series) : Training data for y. 
                X_calib (data frame) : Calibration data for X. 
                y_calib (data series) : Calibration data for y. 
            
            Returns: 
                Returns best estimator with or without calibration applied. 
        """
        best_ = self.best_estimator_b
        best_.fit(X_train, y_train)
        
        if hasattr(best_, "predict_proba"): 

            y_prob = self.best_estimator_b.predict_proba(X_calib)[:, 1]
            true, pred = calibration_curve(y_calib, y_prob)
            plt.plot([0, 1], [0, 1], linestyle = "--")
            plt.plot(pred, true)
            plt.show()

        else: 
            y_prob = self.best_estimator_b.decision_function(X_calib)
            true, pred = calibration_curve(y_calib, y_prob, n_bins= 10)
            plt.plot([0, 1], [0, 1], linestyle = "--")
            plt.plot(pred, true)
            plt.show()
            
         
        if self.calibration_needed == "yes": 
            if self.best_model != "logistic_regression":
                
                if self.best_model == "svm":
                    
                    calibrated = CalibratedClassifierCV(best_, method = "sigmoid", cv = "prefit")
    
                else:
                    calibrated = CalibratedClassifierCV(best_, method = "isotonic", cv = "prefit")
    
                calibrated.fit(X_calib, y_calib)
                self.best_estimator_b = calibrated
            
                return self.best_estimator_b
            
            else: 
                return best_
        else: 
            return best_ 
            
    def predicting_model(self, X, X_train, y_train, X_calib, y_calib):
        

        """
            Predicts with using X return. 

            Parameters: 
                X_train (data frame) : Training data for X. 
                y_train (data series) : Training data for y. 
                X_calib (data frame) : Calibration data for X. 
                y_calib (data series) : Calibration data for y. 
                
            Returns: 
                y_proba (data frame): Data frame that describes probabilities of y. 
                y_test (data series): Data series that indicates whether the entry is 0 or 1. 
        """
        
        calibrated_estimator = self.calibration(X_train, y_train, X_calib, y_calib)
        y_proba = None
        y_test = None
        
        if calibrated_estimator is not None:
            clf = calibrated_estimator          
            if isinstance(clf, CalibratedClassifierCV):
                pipe = clf.estimator          
            else:
                pipe = clf
            
            
            
            prep = pipe.named_steps["preprocessor"]
            X_ = prep.transform(X) 
            last_m = pipe.steps[-1][1]
            if hasattr(last_m, "predict_proba"): 
                
                y_proba = self.best_estimator_b.predict_proba(X)
                y_test = self.best_estimator_b.predict(X)
                m = last_m.predict_proba
                
            else: 
                y_proba = None
                y_test = self.best_estimator_b.predict(X)
                m = last_m.predict
               

            print("Shapley Values ************ ")
            
            
            if hasattr(last_m, "coef_"):
                importances = last_m.coef_[0]
            
            if hasattr(last_m, "feature_importances_"):
                importances = last_m.feature_importances_

            if self.dimensionality_reduction is None: 
                
                feat_names = prep.get_feature_names_out()
                feature_imp = pd.DataFrame({"Column": feat_names, "Importance": importances})
                sorted_imp = feature_imp.sort_values("Importance", ascending=True)
                sorted_imp.plot(x = "Column", y="Importance", kind="barh", figsize=(10,6))
                plt.show()
        
                #cols = prep.get_feature_names_out()
                #shapley_values = shap.Explainer(m, X_, feature_names = cols) 
                #scores = shapley_values(X_) 
                #shap.summary_plot(scores.values[..., 1], X_)

                
                
            return y_proba, y_test

    def analysis(self, X, y, X_train, y_train, X_calib, y_calib):
        """
            Predicts with using X return. 

            Parameters:
                X (data frame): Data for X. 
                y (data frame): Data for y.
                X_train (data frame) : Training data for X. 
                y_train (data series) : Training data for y. 
                X_calib (data frame) : Calibration data for X. 
                y_calib (data series) : Calibration data for y. 
                
            Returns: 
                Returns a confusion Matrix. 
        """

        y_proba, y_test_pred = self.predicting_model(X, X_train, y_train, X_calib, y_calib)

        if y_proba is not None: 
            prediction_l = [1 if proba > self.threshold else 0 for proba in y_proba[:, 1]]
            prediction = np.array(prediction_l)

        else:
            prediction = y_test_pred

        cm = confusion_matrix(y, prediction)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual Class")
        plt.xlabel("Predicted Class")

        plt.show()
        
        

    def store(self, operation=None, file_name = None):
        
        """ 
            A method for storing or loading the model.

            Parameters: 
                operation (str) : Name of the operation. 
                path (str) : Name of the path.

             Returns:
                 Returns loaded model.
        """
        model = self.best_estimator_b 
        if operation == "save": 
            with open(file_name, "wb") as file: 
                pickle.dump(model, file)
            return "Succesfully saved"
            
        elif operation == "load": 
            with open(file_name, "rb") as file: 
                load_model = pickle.load(file)

            self.best_estimator_b = load_model
            return load_model
        else:
            raise ValueError("Please enter save or load ! ")

    


# ## References

# 1- Thrimanne. *Hyperparameter tuning using pipeline end to end ml part 2*. Accessed on March 3, 2025, from https://thrimanne.medium.com/hyperparameter-tuning-using-pipeline-end-to-end-ml-part-2-3-81c68e84d445
#         
# 2- Medium Codex. *Building a mixed type preprocessing pipeline with scikilt learn*. Accessed on March 3, 2025, from https://medium.com/codex/building-a-mixed-type-preprocessing-pipeline-with-scikit-learn-f4d90f5919fa
#   
# 3- Towards Data Science. *Hyperparameter tuning and sampling strategy*. Accessed on February 24, 2025, from https://towardsdatascience.com/hyperparameter-tuning-and-sampling-strategy-1014e05f6c14/
#         
# 4- Kocur, A. *Hyperparameter tuning with pipelines*. Accessed on February 24, 2025, from https://medium.com/@kocur4d/hyper-parameter-tuning-with-pipelines-5310aff069d6
#         
# 5- Kaggle. *Probability Calibration Tutorial*. Accessed on May 6, 2025, from https://www.kaggle.com/code/kelixirr/probability-calibration-tutorial
# 
# 6- Neptune.ai. *Brier score and model calibration*. Accessed on May 6, 2025, from https://neptune.ai/blog/brier-score-and-model-calibration
# 
# 7- Yannawut. *Get column name after fitting the machine learning pipeline*. Acessed on May 11, 2025, from https://yannawut.medium.com/get-column-name-after-fitting-the-machine-learning-pipeline-145a2a8051cc
# 
# 8- Towards Data Science. *Using SHAP values to explain how your machine learning model works*. Accessed on May 11, 2025, from https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137/
# 
# 9- ForecastEgy. *Feature importance in logistic regression*. Accessed on May 6, 2025, from https://forecastegy.com/posts/feature-importance-in-logistic-regression/
# 
# 10- Singh, A. *Hyperparameter tuning beyond the basics*. Accessed on March 7, 2025. https://medium.com/%40abhaysingh71711/hyperparameter-tuning-beyond-the-basics-34d36b014482
