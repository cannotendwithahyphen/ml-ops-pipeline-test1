# In a live coding interview, I would explain this script as follows:
# "This is the deployment part of our MLOps pipeline. We're creating a simple web API using Flask.
# This API will load our trained model and expose an endpoint that can receive new, unseen data and return
# the model's predictions in real-time. This is how we make our model available for other applications to use."

from flask import Flask, request, jsonify
import joblib
import pandas as pd

# "First, we initialize our Flask application."
app = Flask(__name__)

# "Next, we load our trained model. It's important to do this when the application starts,
# so we don't have to reload the model for every prediction request. This is much more efficient."
model = joblib.load('model.pkl')

# "Here, we're defining the 'predict' endpoint. We're specifying that it will accept POST requests,
# which is the standard way to send data to an API."
@app.route('/predict', methods=['POST'])
def predict():
    # "Inside our predict function, the first thing we do is get the JSON data from the request.
    # This data should contain the features for which we want to make a prediction."
    data = request.get_json()

    # "For the model to make a prediction, the input data needs to be in the same format as the data
    # it was trained on. So, we convert the incoming JSON data into a pandas DataFrame."
    features = pd.DataFrame(data['features'])

    # "Now, we use our loaded model to make a prediction on the new data."
    prediction = model.predict(features)

    # "Finally, we need to return the prediction to the user. We'll convert the prediction to a list
    # (in case we're predicting on multiple instances at once) and then return it as a JSON response."
    return jsonify({'prediction': prediction.tolist()})

# "This block allows us to run the Flask development server by simply running `python app.py` in the terminal.
# The `host='0.0.0.0'` makes the server accessible from any IP address, not just localhost."
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
