import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from train_model import actor_columns
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
data = pd.read_csv('IMDb Movies India.csv', encoding='iso-8859-1')

# ... (previous preprocessing code)

# Create mappings for genres, directors, and actors
genre_mapping = {genre: idx for idx, genre in enumerate(data['Genre'].unique())}
director_mapping = {director: idx for idx, director in enumerate(data['Director'].unique())}

actor_mapping = {}
for col in actor_columns:
    actor_mapping[col] = {actor: idx for idx, actor in enumerate(data[col].unique())}

# Save the mappings as CSV files
genre_mapping_df = pd.DataFrame(list(genre_mapping.items()), columns=['Genre', 'NumericValue'])
genre_mapping_df.to_csv('genre_mapping.csv', index=False)

director_mapping_df = pd.DataFrame(list(director_mapping.items()), columns=['Director', 'NumericValue'])
director_mapping_df.to_csv('director_mapping.csv', index=False)

actor_mapping_dfs = []
for col in actor_columns:
    actor_mapping_dfs.append(pd.DataFrame(list(actor_mapping[col].items()), columns=[col, 'NumericValue']))
actor_mapping_combined_df = pd.concat(actor_mapping_dfs, axis=1)
actor_mapping_combined_df.to_csv('actor_mapping.csv', index=False)

# ... (continue with the rest of your preprocessing and training code)
