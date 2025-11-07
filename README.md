# Simple MLOps Pipeline

This project demonstrates a simple, end-to-end MLOps pipeline. It includes scripts for data processing, model training, and a Flask API for serving the model.

## Overview

The pipeline consists of the following components:

- **`data_processing.py`**: Loads the Iris dataset, splits it into training and testing sets, and saves them as CSV files.
- **`train.py`**: Trains a Logistic Regression model on the processed data, evaluates its accuracy, and saves the trained model to a file (`model.pkl`).
- **`app.py`**: A Flask web application that loads the saved model and provides a `/predict` endpoint to make predictions on new data.
- **`run_pipeline.sh`**: A shell script that automates the entire pipeline, from installing dependencies to training the model.
- **`requirements.txt`**: A file listing the Python dependencies for this project.

## Prerequisites

- Python 3.6 or higher
- `pip` for installing Python packages

## How to Run the Pipeline

To run the entire MLOps pipeline, simply execute the `run_pipeline.sh` script from your terminal:

```bash
bash run_pipeline.sh
```

This will:
1. Install the required Python libraries from `requirements.txt`.
2. Run the data processing script to create the training and testing data.
3. Run the model training script to train and save the model.

## How to Use the API

After running the pipeline, you can start the Flask API to serve the model:

```bash
python app.py
```

The API will be running at `http://0.0.0.0:5000`. You can send a POST request to the `/predict` endpoint with new data to get a prediction.

Here is an example of how to do this using `curl`:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "features": [
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 2.9, 4.3, 1.3]
    ]
}' http://localhost:5000/predict
```

The API will return a JSON response with the predictions:

```json
{
  "prediction": [
    0,
    1
  ]
}
```
