# In a live coding interview, I'd introduce this script as follows:
# "This is the second stage of our pipeline: model training. This script takes the data we prepared
# in the previous step, trains a machine learning model, and then saves the trained model to a file.
# This saved model is a critical output of our pipeline, as it's what we'll use for making predictions."

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# "As with the previous script, we'll define a function to keep our code organized."
def train_model():
    # "First, we need to load the training and testing data that we saved earlier.
    # We're using pandas to read the CSV files back into DataFrames."
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv')
    y_test = pd.read_csv('y_test.csv')

    # "Here, we're initializing our model. We've chosen Logistic Regression because it's a simple,
    # well-understood algorithm that's a good starting point for many classification problems.
    # The `random_state=42` is for reproducibility, just like in the data splitting step."
    model = LogisticRegression(random_state=42)

    # "This is the training step. The `fit` method is where the model learns the relationships
    # between our features (X_train) and the target variable (y_train)."
    model.fit(X_train, y_train.values.ravel())

    # "After training, we need to evaluate our model's performance. We use the `predict` method
    # on our test data (X_test) to get the model's predictions."
    predictions = model.predict(X_test)

    # "We then compare these predictions to the actual target values (y_test) to calculate the accuracy.
    # Accuracy is a common metric for classification tasks, and it tells us what percentage of our
    # predictions were correct."
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")

    # "This is a very important step in MLOps: saving the trained model. We use the `joblib` library,
    # which is efficient for saving and loading scikit-learn models. The saved model file ('model.pkl')
    # can then be loaded by other applications, such as a web API, to make predictions on new data."
    joblib.dump(model, 'model.pkl')
    print("Model training complete. Model saved to model.pkl")

# "And again, we use the `if __name__ == '__main__':` block to make our script executable."
if __name__ == '__main__':
    train_model()
