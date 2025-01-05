import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# project root directory
# Go one level up
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load the data
DATASET_PATH = os.path.join(PROJECT_DIR, "dataset.csv")

# Load the data
dataset = pd.read_csv(DATASET_PATH, header=None)
# All columns except the last one
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

print(dataset.head())

# split the data
# 20% for testing and 80% for training
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# calculate accuracy
score = accuracy_score(y_predict, y_test)
print(f"{score * 100:.2f}% of samples were classified correctly!")


# Path for saving the model in the project directory
MODEL_PATH = os.path.join(PROJECT_DIR, "model.p")

# Save the model
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": model}, f)


# functions to draw graphs
def plot_label_distribution(labels):
    # To visualize the count of each label in the dataset.
    # helps to identify class imbalance,
    plt.figure(figsize=(10, 6))
    sns.countplot(labels)
    plt.title("Label Distribution")
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.show()


def plot_accuracy(y_test, y_predict):
    # To display the confusion matrix of the model's predictions
    # helps visualize the model's performance,
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pd.crosstab(y_test, y_predict, rownames=["Actual"], colnames=["Predicted"]),
        annot=True,
        fmt="d",
    )
    plt.title("Accuracy Heatmap")
    plt.show()


# plot graphs
plot_label_distribution(labels)
plot_accuracy(y_test, y_predict)
