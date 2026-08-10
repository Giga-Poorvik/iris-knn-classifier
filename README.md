# Iris KNN Classifier

A beginner-friendly machine-learning project that classifies Iris flowers using a K-Nearest Neighbors (KNN) model built with scikit-learn.

## Overview

The script loads the built-in Iris dataset, splits it into training and test sets, trains a KNN classifier, predicts the species for a new flower measurement, and reports the test accuracy.

## Dataset

The Iris dataset contains 150 flower samples across three species:

- Setosa
- Versicolor
- Virginica

Each sample includes sepal length, sepal width, petal length, and petal width.

## Tech Stack

- Python
- NumPy
- scikit-learn

## Getting Started

### Prerequisites

- Python 3.9 or later

### Installation

```bash
git clone https://github.com/Giga-Poorvik/iris-knn-classifier.git
cd iris-knn-classifier
python -m pip install -r requirements.txt
```

### Run the project

```bash
python iris_knn_classifier.py
```

The program prints a prediction for a sample flower and the model's accuracy on the held-out test set.

## How It Works

1. Load the Iris dataset from scikit-learn.
2. Split the data into training and testing sets.
3. Train a KNN classifier with three neighbors.
4. Predict a new flower's species.
5. Evaluate accuracy on the test data.

## Project Structure

```text
iris-knn-classifier/
├── iris_knn_classifier.py
├── requirements.txt
└── README.md
```

## Future Improvements

- Compare KNN with other classifiers, such as logistic regression and decision trees.
- Visualize the dataset and decision boundaries.
- Add automated tests and model evaluation metrics.
