# Predicting Shopping Preferences
- [Predicting Shopping Preferences](#predicting-shopping-preferences)
  - [1. INTRODUCTION](#1-introduction)
    - [1.1	Project Overview](#11-project-overview)
    - [1.2 Objective](#12-objective)
  - [2.	DATA ANALYSIS](#2-data-analysis)
    - [2.1	Dataset Overview](#21-dataset-overview)
    - [2.2	Unbalanced Dataset](#22-unbalanced-dataset)
  - [3.	DATA PREPERATION AND PREPROCESSING](#3-data-preperation-and-preprocessing)
    - [3.1	Data Loading](#31-data-loading)
    - [3.2	Feature Encoding](#32-feature-encoding)
    - [3.3	Dataset Split](#33-dataset-split)
  - [4.	MODEL DEVELOPMENT](#4-model-development)
    - [4.1	K-Nearest Neighbors (KNN) Classifier](#41k-nearest-neighbors-knn-classifier)
    - [4.2	Cross-Validation](#42-cross-validation)
  - [5.	EVALUATION OF THE MODEL](#5-evaluation-of-the-model)
    - [5.1	Normal Accuracy](#51-normal-accuracy)
    - [5.2	Cross-Validation Accuracy](#52-cross-validation-accuracy)
  - [6.	RESULTS](#6-results)
  - [7.	CONCLUSION](#7-conclusion)

## 1. INTRODUCTION
### 1.1	Project Overview
* The project "Predicting Shopping Preferences" aims to predict individuals' shopping preferences between online and in-store shopping. Data collected through Google Forms includes various features such as age, gender, education, economic status, shopping category, and shopping preference.
### 1.2 Objective
* The primary goal is to develop a predictive model capable of accurately determining shopping preferences based on selected features.
## 2.	DATA ANALYSIS
### 2.1	Dataset Overview
* There are 125 samples in the dataset. Data were collected from many different people via google forms according to the selected features. 
* Dataset consists of age, sex, education, economic status, category and preference. 
* Category includes fashion, book, makeup and care, food and drink, electronics, and home.
* Preference includes in-store and online.

### 2.2	Unbalanced Dataset
* When the data set was analyzed, it was found to have an unbalanced distribution.
* 72% of the dataset is between the ages of 19-26, the majority is female, and about 73% are undergraduates.

## 3.	DATA PREPERATION AND PREPROCESSING
### 3.1	Data Loading
* The dataset is loaded from the "Shopping_preferences.csv" file into a Pandas DataFrame for further analysis.
### 3.2	Feature Encoding
* Categorical variables such as sex, education, economic status, and category are encoded to numerical values to be used as input features for the machine learning model.  One-hot encoding is used.
### 3.3	Dataset Split
* The dataset is split into training and test sets. This division is performed with a 'random_state' value set for repeatability.
    * Test: 80%
    * Train: 20%
## 4.	MODEL DEVELOPMENT
### 4.1	K-Nearest Neighbors (KNN) Classifier
* The KNN classifier is chosen for its simplicity and effectiveness in classification tasks. Also it is suitable for small to medium-sized datasets.
* The number of neighbors (k_neighbors) is set to 3. The choice of K value is important because a smaller 'k' value, such as 3, can make the model sensitive to noise in the data but might capture local patterns well. On the other hand, a larger 'k' can make the model more robust but may lead to a loss of sensitivity to local patterns.
Therefore, the value of k can be chosen by experiment.
* After model and k value selection the model was fitted and trained on the train dataset.
* Then it is predicted on the test  set.

### 4.2	Cross-Validation
* A k-fold cross-validation approach is implemented to evaluate
 the model's performance on different subsets of the data and ensure robustness.
* K-fold is set to 5.
* An accuracy result was reached by averaging all cross-validation accuracy values.

## 5.	EVALUATION OF THE MODEL
### 5.1	Normal Accuracy
The model is initially evaluated using normal accuracy on the test set.
The accuracy_score() function is used within sklearn.metrics library.

### 5.2	Cross-Validation Accuracy
The averaged cross-validation accuracy value was used to evaluate the model and a comparison was made with the normal accuracy value. 
## 6.	RESULTS
* Normal Accuracy Result: 60%
* Cross-Validation Result: 70%


## 7.	CONCLUSION
* In conclusion, the project "Predicting Shopping Preferences" has provided valuable insights into individuals' choices between online and in-store shopping. The model, built on the foundation of the K-Nearest Neighbors algorithm, demonstrated its potential to predict shopping preferences. The results showed that the accuracy was very good. The reason why it was not better was thought to be due to the unbalanced dataset.