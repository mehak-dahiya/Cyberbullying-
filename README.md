# Cyberbullying Detection using ML, DL & Quantum Models

# Description

Comparative analysis of machine learning, deep learning, and quantum models for cyberbullying detection using Hinglish text data.

# Overview

Cyberbullying is a major issue in online platforms, causing serious psychological harm. This project focuses on detecting cyberbullying in Hinglish (Hindi + English) text using multiple computational approaches.

The project evaluates and compares traditional machine learning, deep learning, and quantum-inspired models to identify the most effective method.

# Objectives

* Detect cyberbullying in multilingual text
* Compare ML, DL, and Quantum models
* Improve contextual understanding using BERT embeddings

# Dataset

* Hinglish Cyberbullying Dataset
* Total samples: **18,148**
* Labels:

  * `0` → Non-abusive
  * `1` → Abusive


# Methodology

# Preprocessing

* Lowercasing text
* Removing URLs, mentions, hashtags
* Removing punctuation and numbers
* Handling missing values

# Feature Extraction

* BERT (`bert-base-uncased`) embeddings
* 768-dimensional vector representation

# Models Used

# Machine Learning

* Logistic Regression
* Random Forest
* Support Vector Classifier (SVC)
* Linear SVM
* Naïve Bayes

# Deep Learning

* Bidirectional LSTM (PyTorch)

# Quantum Model

* Quantum-LSTM (PennyLane)


# Results

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| SVC                 | 0.902    | 0.901    |
| Logistic Regression | 0.898    | 0.898    |
| Linear SVM          | 0.896    | 0.896    |
| LSTM                | 0.891    | 0.893    |
| Random Forest       | 0.874    | 0.873    |
| Naïve Bayes         | 0.817    | 0.820    |
| Quantum-LSTM        | 0.642    | 0.503    |


# Output Visualizations

* Class distribution graph
* Word cloud
* Confusion matrices
* Model comparison graph

# Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* PyTorch
* Transformers (BERT)
* PennyLane
* Matplotlib, Seaborn

#  Insights

* SVC achieved highest accuracy (0.902)
* Classical ML models performed strongly
* BERT improved contextual understanding
* Quantum model needs further optimization


# Future Scope

* Real-time detection system
* Web/app integration
* Multilingual expansion
* Improved quantum models


