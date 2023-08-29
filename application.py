from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the list of feature names used in training
with open('feature_names.pkl', 'rb') as feature_names_file:
    feature_names = pickle.load(feature_names_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(request.form['year']),
                      float(request.form['votes']),
                      float(request.form['genre']),
                      float(request.form['director']),
                      float(request.form['actor1']),
                      float(request.form['actor2']),
                      float(request.form['actor3'])]

    # Create a DataFrame with the input features
    input_df = pd.DataFrame([input_features], columns=feature_names)

    # Use the trained model to make a prediction
    prediction = model.predict(input_df)

    return render_template('index.html', prediction_text='Predicted Rating: {:.2f}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
