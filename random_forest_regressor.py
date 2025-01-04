import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger les données
data = pd.read_csv("database_final.csv")

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

X = data[features]
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
model = RandomForestRegressor(n_estimators=21, max_depth=10, criterion='squared_error',random_state=42)
model.fit(X_train, y_train)

# 5. Évaluer le modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Erreur quadratique moyenne :", rmse)
print("R2 Score :",  r2_score(y_test, y_pred))

# Affichage des importances des caractéristiques
importances = model.feature_importances_
encoded_features = encoder.get_feature_names_out(['Chaîne', 'Jour'])
# Ajouter les colonnes binaires des genres à la liste des caractéristiques
genres_features = genres_df.columns.tolist()
nationalite_features = national_df.columns.to_list()
numeric_features = ['Durée (en min.)', 'IMDB - Note moyenne', 'Mois', 'Vacances scolaires', 'IMDB - Nombre de votes', 'Année de sortie', 'Année de diffusion', 'Week-end', 'Saison']
feature_names = np.concatenate([numeric_features, encoded_features , genres_features, nationalite_features])

# Triez les colonnes par importance
indices = np.argsort(importances)[::-1]

# Affichez les importances
plt.figure(figsize=(10, 6))
plt.title("Importance des caractéristiques")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.tight_layout()
#plt.show()

# Prédire de nouvelles données
new_data = pd.DataFrame([
    # Le comte de Monte Cristo
    {
        'Chaîne': 'M6',
        'Genres': 'Action,Adventure,Drama',
        'Nationalité': 'FRANCE',
        'Durée (en min.)': 178,
        'IMDB - Note moyenne': 7.7,
        'IMDB - Nombre de votes': 21357,
        'Année de sortie': 2024,
        'Année de diffusion': 2025,
        'Jour': 'Saturday',
        'Mois': 4,
        'Vacances scolaires': 'oui',
        'Week-end': 1,
        'Saison': 2
    },
    # L'amour ouf
    {
        'Chaîne': 'France 2',
        'Genres': 'Crime,Drama,Romance',
        'Nationalité': 'FRANCE',
        'Durée (en min.)': 166,
        'IMDB - Note moyenne': 7.3,
        'IMDB - Nombre de votes': 2385,
        'Année de sortie': 2024,
        'Année de diffusion': 2025,
        'Jour': 'Wednesday',
        'Mois': 12,
        'Vacances scolaires': 'oui',
        'Week-end': 0,
        'Saison': 1
    },
    # Un p'tit truc en plus
    {
        'Chaîne': 'M6',
        'Genres': 'Comedy',
        'Nationalité': 'FRANCE',
        'Durée (en min.)': 99,
        'IMDB - Note moyenne': 7.0,
        'IMDB - Nombre de votes': 3870,
        'Année de sortie': 2024,
        'Année de diffusion': 2025,
        'Jour': 'Saturday',
        'Mois': 9,
        'Vacances scolaires': 'non',
        'Week-end': 1,
        'Saison': 3
    }
])

# Séparer les genres pour appliquer l'encodage MultiLabelBinarizer
new_data['Genres'] = new_data['Genres'].apply(lambda x: x.split(','))

# Séparer les nationalités par des slash
new_data['Nationalité'] = new_data['Nationalité'].apply(lambda x: [i.strip() for i in x.split('/')] if isinstance(x, str) else [])

# Encoder les nouvelles données (exemple d'utilisation des encodeurs et scalers)
new_data_encoded = encoder.transform(new_data[['Chaîne', 'Jour']])

# Encodage de la colonne "Vacances scolaires"
new_data['Vacances scolaires'] = vacances_encoder.transform(new_data['Vacances scolaires'])

# Normalisation des données numériques
new_data_scaled = scaler.transform(new_data[['Durée (en min.)', 'IMDB - Note moyenne', 'IMDB - Nombre de votes', 'Année de sortie', 'Année de diffusion', 'Mois', 'Week-end', 'Saison']])

# Encoder les genres avec MultiLabelBinarizer
genres_encoded = mlb.transform(new_data['Genres'])

# Encoder les nationalités
nationalite_encoded = mlb_nationalite.transform(new_data['Nationalité'])

# Combiner toutes les features des nouvelles données
new_data_final = np.hstack([new_data_scaled, new_data_encoded.toarray(), new_data[['Vacances scolaires']].values, genres_encoded, nationalite_encoded])

# Prédire avec le modèle
new_predictions = model.predict(new_data_final)

# Affichage des résultats
print("Prédiction pour le Comte de Monte Cristo diffusé sur M6 un samedi soir d'avril 2025 pendant les vacances :", new_predictions[0], "millions de téléspectateurs.")
print("Prédiction pour l'Amour OUF diffusé sur France 2 un mercredi soir de décembre 2025 pendant les vacances :", new_predictions[1], "millions de téléspectateurs.")
print("Prédiction pour un p'tit truc en plus diffusé sur M6 un samedi soir de septembre 2025 hors vacances :", new_predictions[2], "millions de téléspectateurs.")
