# Intelligent-ML-DL-Pipeline-for-Classification-Clustering-and-NLP

This project presents an end-to-end data science pipeline combining traditional machine learning, deep learning, and natural language processing (NLP) techniques.

For structured data analysis, the project focuses on classifying breast cancer tumours as benign or malignant using both supervised and unsupervised learning approaches. The primary dataset used is the Breast Cancer Wisconsin Dataset:
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

To better simulate real-world scenarios, the dataset was intentionally modified by introducing noise, including missing values and class imbalance. This enabled the practical application of preprocessing techniques such as median imputation, SMOTE for handling imbalance, and feature scaling.

A Random Forest classifier was implemented for supervised learning, while K-Means and DBSCAN clustering algorithms were applied for unsupervised pattern exploration. Principal Component Analysis (PCA) was used for dimensionality reduction and visualisation.

In addition, the project extends to NLP-based classification using a real-world dataset of job postings:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

For the NLP task, deep learning models including LSTM and BiLSTM architectures were developed to classify job postings as real or fraudulent. Text preprocessing steps included tokenisation, stopword removal, and the use of GloVe embeddings to capture semantic meaning.

Model performance across all approaches was evaluated using accuracy, precision, recall, F1-score, and cross-validation, providing a comprehensive comparison between traditional machine learning and deep learning techniques.
