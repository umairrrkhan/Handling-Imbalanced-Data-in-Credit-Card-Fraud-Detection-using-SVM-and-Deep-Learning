# Handling Imbalanced Data in Credit Card Fraud Detection using SVM and Deep Learning

## Overview

This project applies ML models like SVM and Neural Networks for credit card fraud detection on highly imbalanced data. Resampling techniques like oversampling and SMOTE are used to handle the skewed class distribution.

The goal is to effectively identify rare fraud cases from the imbalanced dataset.

## Data 

The credit card dataset is obtained from Kaggle:
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

It contains 284,807 transactions with 30 features. Only 492 (0.172%) are fraud cases. This poses a challenge.

## Usage

Implemented in Python with:

- Pandas, NumPy
- Scikit-learn
- Keras, TensorFlow
- Logistic regression
- SVM

To use:

1. Clone the repo  
2. Install requirements
3. Run Jupyter notebooks

## Methodology

Steps include:
  
- Exploring imbalance
- Preprocessing  
- Baseline models - LR, SVM, Random Forest 
- Resampling techniques 
- Tuning hyperparameters
- Evaluation metrics - precision f1 score 
- Feature analysis
- Test set evaluation

## Results  

The neural network model with SMOTE achieved the best performance - ROC AUC of 0.95 and F1 score of 0.83. Top features were V10-V28 transaction amounts. Oversampling balanced class distributions.

## Future Work

Further improvements through:

- Advanced neural network architectures  
- Class weights and thresholds
- Anomaly detection 
- Analysis of sequences

This provides a benchmark using common models and resampling techniques.

## Contact 

Email - umairh1819@gmail.com
