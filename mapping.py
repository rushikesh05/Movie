from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from train_model import data , actor_columns


# Assuming data is the loaded dataset
label_encoder = LabelEncoder()
data['Genre'] = label_encoder.fit_transform(data['Genre'])

genre_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
# print(genre_mapping)
actor_mapping = {}
for col in actor_columns:
    label_encoder = LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col])
    actor_mapping[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

print(actor_mapping)
