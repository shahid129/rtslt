import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# load the dataset
dataset = pd.read_csv("dataset.csv", header=None)
# get all columns except the last one for features
features = dataset.iloc[:, :-1].values
# get the last column for labels
labels = dataset.iloc[:, -1].values

# split the data into training and testing sets
# 20% for testing and 80% for training
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, shuffle=True, stratify=labels
)

# load the trained model
with open("model.p", "rb") as model_file:
    model_data = pickle.load(model_file)
hand_sign_model = model_data["model"]


# checking data quality
def check_data_quality(features, labels):
    print("check the data quality:")

    # check if there are any missing values in the features or labels
    print(f"Missing values in features: {np.isnan(features).sum()}")
    print(f"Missing values in labels: {np.isnan(labels).sum()}")

    # look at the distribution of the labels
    print("\nHow the labels are distributed:")
    print(pd.Series(labels).value_counts())
    print()


# evaluate model performance
def evaluate_model_performance(model, x_test, y_test):
    print("Evaluating how well the model performs:")

    # make predictions on the test set
    y_pred = model.predict(x_test)

    # calculate the accuracy and show the classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print()


# check for overfitting and underfitting
def check_overfitting_underfitting(model, x_train, y_train, x_test, y_test):
    print("Checking for overfitting and underfitting:")

    # evaluate the model on both training and testing data
    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)

    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing accuracy: {test_accuracy * 100:.2f}%")

    # compare training and testing accuracies to
    # identify overfitting or underfitting
    if train_accuracy > test_accuracy:
        print("The model is overfitting the training data.")
    elif train_accuracy < test_accuracy:
        print("The model isunderfitting the data.")
    else:
        print("The model has good balance between training and testing.")
    print()


# plotting the label distribution
def plot_label_distribution(labels):
    plt.figure(figsize=(10, 6))
    sns.countplot(labels)
    plt.title("Distribution of Labels")
    plt.xlabel("Labels")
    plt.ylabel("Frequency")
    plt.show()


# running all the tests
check_data_quality(features, labels)
evaluate_model_performance(hand_sign_model, x_test, y_test)
check_overfitting_underfitting(
    hand_sign_model, x_train, y_train, x_test, y_test)
plot_label_distribution(labels)
