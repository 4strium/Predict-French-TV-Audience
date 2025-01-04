import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger les données
data = pd.read_csv("database_final.csv")

# 2. Prétraitement
data['Date de diffusion'] = pd.to_datetime(data['Date de diffusion'])
data['Jour'] = data['Date de diffusion'].dt.day_name()
data['Mois'] = data['Date de diffusion'].dt.month

# Sélectionner les colonnes nécessaires
features = ['Chaîne', 'Genres', 'Durée (en min.)', 'IMDB - Note moyenne', 'IMDB - Nombre de votes', 'Jour', 'Mois', 'Vacances scolaires']
target = 'Téléspectateurs (en millions)'

X = data[features]
y = data[target]

# Encoder les variables catégoriques
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X[['Chaîne', 'Genres', 'Jour']])

# Normaliser les colonnes numériques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['Durée (en min.)', 'IMDB - Note moyenne', 'IMDB - Nombre de votes', 'Mois']])

# Convertir "Vacances scolaires" en variable binaire
vacances_encoder = LabelEncoder()
X['Vacances scolaires'] = vacances_encoder.fit_transform(X['Vacances scolaires'])

# Combiner toutes les features
X_final = np.hstack([X_scaled, X_encoded.toarray(), X[['Vacances scolaires']].values])

# 3. Séparer en jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# 4. Modélisation
model = KNeighborsRegressor(n_neighbors=5,weights='uniform')
model.fit(X_train, y_train)

# 5. Évaluer le modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Erreur quadratique moyenne :", rmse)
print("R2 Score :",  r2_score(y_test, y_pred))

# Prédire de nouvelles données
new_data = pd.DataFrame({
    'Chaîne': ['M6'],
    'Genres': ['Action,Adventure,Drama'],
    'Durée (en min.)': [178],
    'IMDB - Note moyenne': [7.7],
    'IMDB - Nombre de votes' : [21357],
    'Jour': ['Saturday'],
    'Mois': [4],
    'Vacances scolaires': ['non']
})

# Encoder les nouvelles données
new_data_encoded = encoder.transform(new_data[['Chaîne', 'Genres', 'Jour']])

new_data['Vacances scolaires'] = vacances_encoder.transform(new_data['Vacances scolaires'])

# Normaliser les nouvelles données
new_data_scaled = scaler.transform(new_data[['Durée (en min.)', 'IMDB - Note moyenne', 'IMDB - Nombre de votes', 'Mois']])

# Combiner toutes les features des nouvelles données
new_data_final = np.hstack([new_data_scaled, new_data_encoded.toarray(), new_data[['Vacances scolaires']].values])

# Prédire avec le modèle
new_predictions = model.predict(new_data_final)
print("Prédiction pour le film que vous proposez :", new_predictions[0],"millions de téléspectateurs.")
