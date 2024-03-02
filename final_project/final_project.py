import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
# Load the dataset
data = pd.read_csv("Shopping_preferences.csv")

# Data preprocessing
class_label_encoder = {label: i for i, label in enumerate(data['Preference'].unique())}
data['Preference'] = data['Preference'].map(class_label_encoder)

X = data[['Age', 'Sex', 'Education', 'Economic Status', 'Category']]
y = data['Preference']

# Convert categorical variables to numerical
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

k_neighbors = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k_neighbors)


knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

# Evaluate the model with normal accuracy
normal_accuracy = accuracy_score(y_test, y_pred)
print(f"Normal Accuracy: {normal_accuracy * 100:.2f}%")

def cross_validate(X, y, k=5):
    fold_size = len(X) // k
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    accuracies = []

    for i in range(k):
        validation_indices = indices[i * fold_size: (i + 1) * fold_size]
        training_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        X_train, y_train = X.iloc[training_indices], y.iloc[training_indices]
        X_val, y_val = X.iloc[validation_indices], y.iloc[validation_indices]

        knn_classifier = KNeighborsClassifier(n_neighbors=k_neighbors)
        knn_classifier.fit(X_train, y_train)

        y_pred = knn_classifier.predict(X_val)

        accuracy = np.mean(y_pred == y_val)
        accuracies.append(accuracy)

    return np.mean(accuracies)

# Select whole dataset for cross-validation
X_full = X
y_full = y

#Evaluate the model with cross validation
k_fold = 5  
cross_val_accuracy = cross_validate(X_full, y_full, k=k_fold)
print(f"Cross-Validation Accuracy (k={k_fold}): {cross_val_accuracy * 100:.2f}%")
