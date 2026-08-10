"""Train and evaluate a K-Nearest Neighbors classifier on the Iris dataset."""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    """Train the model and print a sample prediction and test accuracy."""
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)

    sample = np.array([[5.0, 2.9, 1.0, 0.2]])
    prediction = model.predict(sample)[0]
    accuracy = model.score(x_test, y_test)

    print(f"Sample prediction: {iris.target_names[prediction]}")
    print(f"Test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
