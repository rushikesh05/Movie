import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
data = pd.read_csv('IMDb Movies India.csv', encoding='iso-8859-1')

# Drop rows with missing values
data.dropna(inplace=True)

# Convert Votes column to integer
data['Votes'] = data['Votes'].str.replace(',', '').astype(int)

# Create a label encoder for categorical features
label_encoder = LabelEncoder()

# Apply label encoding to categorical columns
data['Genre'] = label_encoder.fit_transform(data['Genre'])
data['Director'] = label_encoder.fit_transform(data['Director'])

# Extract and label encode actor names
actor_columns = ['Actor 1', 'Actor 2', 'Actor 3']
for col in actor_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Create a new DataFrame with extracted features
features = data.drop(columns=['Rating', 'Duration', 'Name'])  # Features
target = data['Rating']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize a Random Forest Regressor model
random_forest_reg = RandomForestRegressor()

# Train the model
random_forest_reg.fit(X_train, y_train)

# ... (previous code for preprocessing and training)

# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(random_forest_reg, model_file)

# Save the list of feature names used in training
with open('feature_names.pkl', 'wb') as feature_names_file:
    pickle.dump(features.columns.tolist(), feature_names_file)
