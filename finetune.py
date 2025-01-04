import xgboost.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
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

# 4. Définir le modèle et les paramètres pour la recherche
model = xgboost.XGBRegressor(tree_method='hist', device='cuda', seed=42)

param_grid = {
    'n_estimators': [28,29,30,31,32],
    'max_depth': [3],
    'learning_rate': [0.35,0.38,0.4,0.42,0.45],
    'subsample': [0.95,0.98, 1.0],
    'colsample_bytree': [0.75, 0.8, 0.85]
}

# 5. Recherche de grille
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Meilleurs paramètres et évaluation
best_model = grid_search.best_estimator_
print("Meilleurs paramètres :", grid_search.best_params_)

# 6. Évaluer le modèle avec les meilleurs paramètres
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Erreur quadratique moyenne :", rmse)

# Affichage des importances des caractéristiques
importances = best_model.feature_importances_
encoded_features = encoder.get_feature_names_out(['Chaîne', 'Genres', 'Jour'])
numeric_features = ['Durée (en min.)', 'IMDB - Note moyenne', 'Mois', 'Vacances scolaires', 'IMDB - Nombre de votes']
feature_names = np.concatenate([numeric_features, encoded_features])

# Triez les colonnes par importance
indices = np.argsort(importances)[::-1]

# Affichez les importances
plt.figure(figsize=(10, 6))
plt.title("Importance des caractéristiques")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.tight_layout()
# plt.show()
