import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger les données
data = pd.read_csv("database_final.csv")

# 2. Prétraitement
data['Date de diffusion'] = pd.to_datetime(data['Date de diffusion'])
data['Jour'] = data['Date de diffusion'].dt.day_name()
data['Mois'] = data['Date de diffusion'].dt.month

# Sélectionner les colonnes nécessaires
features = ['Chaîne', 'Genres', 'Durée (en min.)', 'IMDB - Note moyenne', 'Jour', 'Mois', 'Vacances scolaires']
target = 'Téléspectateurs (en millions)'

X = data[features]
y = data[target]

# Encoder les variables catégoriques
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X[['Chaîne', 'Genres', 'Jour']])

# Normaliser les colonnes numériques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['Durée (en min.)', 'IMDB - Note moyenne', 'Mois']])

# Combiner toutes les features
X_final = np.hstack([X_scaled, X_encoded.toarray()])

# 3. Séparer en jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# 4. Modélisation
model = DecisionTreeRegressor(max_depth=6, random_state=42, criterion="squared_error")
model.fit(X_train, y_train)

# 5. Évaluer le modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Erreur quadratique moyenne :", rmse)

importances = model.feature_importances_
# Obtenez les noms des colonnes encodées
encoded_features = encoder.get_feature_names_out(['Chaîne', 'Genres', 'Jour'])
numeric_features = ['Durée (en min.)', 'IMDB - Note moyenne', 'Mois']
feature_names = np.concatenate([numeric_features, encoded_features])

# Triez les colonnes par importance
indices = np.argsort(importances)[::-1]

# Affichez les importances
plt.figure(figsize=(10, 6))
plt.title("Importance des caractéristiques")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()
