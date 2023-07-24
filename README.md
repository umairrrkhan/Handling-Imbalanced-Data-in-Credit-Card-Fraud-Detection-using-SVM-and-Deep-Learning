# Handling Imbalanced Data in Credit Card Fraud Detection using SVM and Deep Learning

## Introduction

Credit card fraud detection is an important and challenging real-world problem that involves significant financial losses. Machine learning provides effective approaches for identifying fraudulent transactions. However, frauds occur rarely compared to legitimate transactions, creating highly imbalanced datasets. Standard algorithms struggle to detect the minority positive class of frauds.

This project focuses on evaluating machine learning techniques for credit card fraud detection on imbalanced data. Models like logistic regression, SVM, random forest, and neural networks are tested. To handle the disproportionate class distribution

The algorithms and resampling techniques are optimized to identify rare cases of fraud despite the skewed dataset precision-recall curve, and F1-score are utilized for evaluation. In-depth analysis provides insights into significant patterns and effective approaches.

This project provides an extensive benchmark for applying machine learning to the critical and timely problem of fraud detection in imbalanced financial data. The techniques and findings will be broadly useful for related domains involving identification of rare classes.

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
