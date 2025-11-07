#!/bin/bash

# In a live coding interview, I'd explain this script like this:
# "This is a simple shell script that automates our entire MLOps pipeline.
# The `set -e` command at the top is important for automation. It means that if any command fails,
# the script will exit immediately. This prevents the pipeline from continuing in an inconsistent state."
set -e

# "First, we'll print a message to the console to indicate that the pipeline has started."
echo "Starting the MLOps pipeline..."

# "The first step in our pipeline is to install the necessary Python libraries.
# The `pip install -r requirements.txt` command reads our `requirements.txt` file and installs
# all the specified libraries. This ensures that our pipeline runs in a consistent environment."
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt

# "Once the dependencies are installed, we run our data preparation script.
# This script will load the raw data, process it, and save the training and testing sets to CSV files."
echo "Step 2: Running data processing..."
python data_processing.py

# "After the data has been prepared, we run our model training script.
# This script will load the processed data, train the model, and save the trained model to a file."
echo "Step 3: Running model training..."
python train.py

# "Finally, we'll print a success message to indicate that the pipeline has completed successfully."
echo "MLOps pipeline completed successfully."
