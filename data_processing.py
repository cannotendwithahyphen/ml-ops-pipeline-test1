# In a live coding interview, I would start by explaining the purpose of this script.
# "This script is responsible for the first step in our MLOps pipeline: data preparation.
# We need to load our data, process it, and then save it in a format that our model training script can use.
# For this demonstration, we're using the classic Iris dataset from scikit-learn, which is a common choice for examples like this."

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# "First, we'll define a function to encapsulate our data processing logic.
# This makes our code more modular and easier to test."
def prepare_data():
    # "Here, we're loading the Iris dataset. It's a simple, clean dataset, which is perfect for this example.
    # The data is returned as a Bunch object, which is similar to a dictionary."
    iris = load_iris()

    # "Next, we convert the data into a pandas DataFrame. This is a very common step in data science workflows
    # as DataFrames are incredibly versatile for data manipulation and analysis."
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # "We also need to add the target variable (the species of iris) to our DataFrame."
    data['target'] = iris.target

    # "Now, we'll split our data into features (X) and the target (y)."
    X = data.drop('target', axis=1)
    y = data['target']

    # "This is a crucial step: splitting the data into training and testing sets.
    # We use the training set to train our model, and the testing set to evaluate its performance on unseen data.
    # The `test_size=0.2` means we're using 20% of the data for testing, and `random_state=42` ensures that
    # the split is the same every time we run the script, which is important for reproducibility."
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # "Finally, we save our processed data to CSV files. This is a common way to pass data between different
    # stages of a pipeline. Our training script will then load these files."
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    # "It's good practice to print a confirmation message to the console."
    print("Data processing complete. Files saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")

# "The `if __name__ == '__main__':` block is standard Python practice. It ensures that the `prepare_data`
# function is called only when the script is executed directly, not when it's imported as a module."
if __name__ == '__main__':
    prepare_data()
