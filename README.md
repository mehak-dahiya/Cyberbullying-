
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

Outputs
<img width="600" height="500" alt="SVC_confusion_matrix" src="https://github.com/user-attachments/assets/509048e0-234a-4e2f-b2fd-eec22e59e710" />
<img width="600" height="500" alt="Random_Forest_confusion_matrix" src="https://github.com/user-attachments/assets/295fd722-773c-47c4-b17f-56fda49e3710" />
<img width="600" height="500" alt="Quantum-LSTM_Classifier_confusion_matrix" src="https://github.com/user-attachments/assets/61e1cc6f-8649-4d30-a8ed-f25a77e67116" />
<img width="600" height="500" alt="Quantum_RNN_confusion_matrix" src="https://github.com/user-attachments/assets/d8fd7ef1-e9b6-4e25-95f7-bb8d670c3993" />
<img width="600" height="500" alt="Naive_Bayes_confusion_matrix" src="https://github.com/user-attachments/assets/c56d3a89-6068-49a3-9ee7-1e7205140422" />
<img width="600" height="500" alt="LSTM_Classifier_confusion_matrix" src="https://github.com/user-attachments/assets/f5116ac8-ee56-4e29-9ef1-35d6da0ef38d" />
<img width="600" height="500" alt="Logistic_Regression_confusion_matrix" src="https://github.com/user-attachments/assets/22520530-bba3-4bd1-9e42-db90c345b050" />
<img width="600" height="500" alt="Linear_SVM_confusion_matrix" src="https://github.com/user-attachments/assets/361397e1-372b-4fff-abe4-a17bf31e5e48" />
<img width="600" height="400" alt="class_count" src="https://github.com/user-attachments/assets/e63523a2-6964-4378-8a53-1cae301beb5c" />
<img width="600" height="500" alt="BERT_Pipeline_confusion_matrix" src="https://github.com/user-attachments/assets/4612fa67-e05d-4577-950a-5d1ac009e1d6" />
<img width="1000" height="600" alt="model_performance_comparison" src="https://github.com/user-attachments/assets/ac555fd3-9529-4053-8445-5d3b2178f8e1" />
<img width="640" height="480" alt="wordcloud" src="https://github.com/user-attachments/assets/68dea3ef-04db-4210-911d-546b57b52152" />


