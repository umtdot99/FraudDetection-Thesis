# Implementation of Fraud Detection Techniques
**Master’s Thesis | KU Leuven**

A flexible, modular, and customizable framework for Fraud Detection. This architecture allows users to supply any dataset with a binary target and supports Supervised Learning, Unsupervised Learning, and advanced Neural Network architectures.

---

## 📌 Overview
This framework provides a reusable approach for the fraud detection lifecycle, supporting:
* **Customizable Pre-processing:** User-defined pipelines for data cleaning and transformation.
* **Imbalance Handling:** Integrated tools to manage highly imbalanced datasets (e.g., Undersampling, SMOTE).
* **Advanced Modeling:** Supports Classical ML, Autoencoders, CNNs, and Generative Adversarial Networks (GANs).
* **Automated Tuning:** Built-in support for hyperparameter optimization for classical algorithms.
* **Research Benchmarking:** A structured environment to compare custom methods against state-of-the-art models.

## 📊 Dataset
While the framework is dataset-agnostic, experiments were conducted using the **Credit Card Fraud Detection** dataset from Kaggle:
* **Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
* **Stats:** 284,807 transactions; 492 fraud cases (0.172% class imbalance).
* **Features:** Includes PCA-transformed variables (V1-V28), Time, and Amount.

## 🛠 Requirements
The environment was managed using Anaconda. To replicate the environment and dependencies:

```bash
conda env create -f environment.yml
conda activate [env_name]
```

##🚀 Usage Example
The following snippet demonstrates how to initialize the data, handle imbalance, and execute the pipelines:
```python
# 1. Initialize Data
data = DataGathering(file_path="creditcard.csv", target="Class", 
                     pipeline_numerical=pipe_num, pipeline_categorical=pipe_cat)

# 2. Handle Class Imbalance
imbalance_handler = ImbalanceHandler(method="undersampling")

# 3. Training a Neural Network
neunet = NeuralN(input_dimension=inp_dim, output_dimension=out_dim, 
                 hidden_layers=hidden, loss_method="BCEwLogit", 
                 opt_method="Adam", lr=0.001, epochs=10)

model = FraudDetectionPipeline(data=data, imbalance_handler=imbalance_handler, neural_networks=neunet)
model.run_neuralnetwork()

# 4. Running Supervised Learning Models
trainer = TrainModel(search_strategy="grid", param_grid=PARAM_GRID, cv=3, 
                     model_dictionary=MODEL_DICT, threshold=0.45)

model = FraudDetectionPipeline(data=data, imbalance_handler=imbalance_handler, trainer=trainer)
model.run_supervisedmodels()
```

## 🧩 Extensibility
The pipeline is designed with a modular architecture to allow for:

Custom Loss Functions: Implementation of specialized loss metrics for anomaly detection.

New Architectures: Easy integration of additional DL models or class imbalance techniques.

Interpretability: Space for implementing SHAP/LIME or other feature importance techniques.

## 🙏 Acknowledgements
I would like to express my deepest gratitude to my supervisor Prof. Dr. Wouter Verbeke and daily advisor Dr. Bruno Deprez for their guidance. Full references and bibliography are provided in the complete thesis document.
