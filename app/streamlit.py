import streamlit as st
import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Charger les données
data = pd.read_csv("database.csv")

# 2. Prétraitement
data['Date de diffusion'] = pd.to_datetime(data['Date de diffusion'])
data['Mois'] = data['Date de diffusion'].dt.month
data['Week-end'] = data['Jour'].isin(['Saturday', 'Sunday']).astype(int)
data['Saison'] = data['Date de diffusion'].dt.month % 12 // 3 + 1  # 1: hiver, 2: printemps, 3: été, 4: automne
data['Année de diffusion'] = data['Date de diffusion'].dt.year
data['Nationalité'] = data['Nationalité'].str.upper()

# Séparer les genres par des virgules
data['Genres'] = data['Genres'].apply(lambda x: x.split(','))
# Appliquer MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(data['Genres'])

# Créer un DataFrame avec les genres encodés
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# Séparer les nationalités par des slash
data['Nationalité'] = data['Nationalité'].apply(lambda x: [i.strip() for i in x.split('/')] if isinstance(x, str) else [])

mlb_nationalite = MultiLabelBinarizer()
nationalite_encoded = mlb_nationalite.fit_transform(data['Nationalité'])

# Créer un DataFrame avec les nationalités encodées
national_df = pd.DataFrame(nationalite_encoded, columns=mlb_nationalite.classes_)

# Sélectionner les colonnes nécessaires
features = ['Chaîne', 'Genres', 'Nationalité', 'Durée (en min.)', 'IMDB - Note moyenne', 'IMDB - Nombre de votes', 'Année de sortie', 'Jour', 'Mois', 'Année de diffusion', 'Vacances scolaires', 'Week-end', 'Saison']
target = 'Téléspectateurs (en millions)'

X = data[features].copy()
y = data[target]

# Encoder les variables catégoriques
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X[['Chaîne', 'Jour']])

# Normaliser les colonnes numériques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['Durée (en min.)', 'IMDB - Note moyenne', 'IMDB - Nombre de votes', 'Année de sortie', 'Année de diffusion', 'Mois', 'Week-end', 'Saison']])

# Convertir "Vacances scolaires" en variable binaire
vacances_encoder = LabelEncoder()
X['Vacances scolaires'] = vacances_encoder.fit_transform(X['Vacances scolaires'])

# Combiner toutes les features
X_final = np.hstack([X_scaled, X_encoded.toarray(), X[['Vacances scolaires']].values, genres_df.values, national_df.values])

# 3. Séparer en jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# 4. Modélisation
model = xgboost.XGBRegressor(max_depth=3, subsample=0.91, tree_method='hist', seed=42, n_estimators=30, learning_rate=0.32)
model.fit(X_train, y_train)

# 5. Évaluer le modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
