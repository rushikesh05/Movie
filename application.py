from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the mappings for actors, genres, and directors
with open('actor_mapping.pkl', 'rb') as actor_mapping_file:
    actor_mapping = pickle.load(actor_mapping_file)

with open('genre_mapping.pkl', 'rb') as genre_mapping_file:
    genre_mapping = pickle.load(genre_mapping_file)

with open('director_mapping.pkl', 'rb') as director_mapping_file:
    director_mapping = pickle.load(director_mapping_file)

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
    prediction = model.predict(input_df)[0]

    # Interpret numeric values using mappings
    genre_name = genre_mapping[int(request.form['genre'])]
    director_name = director_mapping[int(request.form['director'])]
    actor1_name = actor_mapping['Actor 1'][int(request.form['actor1'])]
    actor2_name = actor_mapping['Actor 2'][int(request.form['actor2'])]
    actor3_name = actor_mapping['Actor 3'][int(request.form['actor3'])]

    return render_template('index.html', prediction_text='Predicted Rating: {:.2f}'.format(prediction),
                           genre_name=genre_name, director_name=director_name,
                           actor1_name=actor1_name, actor2_name=actor2_name, actor3_name=actor3_name)
if __name__ == '__main__':
    app.run(debug=True)
