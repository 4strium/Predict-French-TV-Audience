import streamlit as st
import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from vacances_scolaires_france import SchoolHolidayDates
import datetime
import requests

# Affichage 
st.title("French TV Audience Prediction")

@st.cache_data
def load_data():
  return pd.read_csv("database.csv")

data = load_data()

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

with st.empty():
  st.write("🧠 Training in progress...")
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
  st.write(":material/check: Model trained !")

# 5. Évaluer le modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

st.subheader("Model infos :")
st.write("Erreur quadratique moyenne :", rmse)
st.write("R2 Score :",  r2_score(y_test, y_pred))

st.subheader("Predict your film :")
imdb_id = st.text_input("IMDB Film ID (ttxxxxx)", max_chars=14)
channel = st.selectbox("Channel", ["TF1", "France 2", "France 3", "France 4", "France 5", "M6", "Arte", "C8", "W9", "TMC", "TFX", "TF1 Séries Films", "6ter", "Gulli", "Canal +", "C Star", "NRJ12", "Chérie 25"])
date_diffusion = st.date_input("Broadcast date", format="DD/MM/YYYY")

# Prédire avec le modèle
if st.button("Predict !") : 
  akas_json = requests.get(f"https://api.imdbapi.dev/titles/{imdb_id}/akas").json()
  title_france = next((aka["text"] for aka in akas_json["akas"] if aka["country"]["code"] == "FR"), None)

  all_data_json = requests.get(f"https://api.imdbapi.dev/titles/{imdb_id}").json()
  runtime_minutes = all_data_json['runtimeSeconds'] // 60
  country = all_data_json['originCountries'][0]['name']
  genres = ','.join(all_data_json['genres'])
  average_rating = all_data_json['rating']['aggregateRating']
  num_votes = all_data_json['rating']['voteCount']
  year_release = all_data_json['startYear']

  film_to_predict = {
    'TITRE' : title_france,
    'Chaîne': channel,
    'Genres': genres,
    'Nationalité': country,
    'Durée (en min.)': runtime_minutes,
    'IMDB - Note moyenne': float(average_rating),
    'IMDB - Nombre de votes': int(num_votes),
    'Année de sortie': int(year_release),
    'Année de diffusion': date_diffusion.year,
    'Jour': pd.Timestamp(date_diffusion).day_name(),
    'Mois': date_diffusion.month,
    'Vacances scolaires': 'oui' if SchoolHolidayDates().is_holiday(datetime.date(date_diffusion.year, date_diffusion.month, date_diffusion.day)) else 'non',
    'Week-end': 1 if pd.Timestamp(date_diffusion).day_name() in ['Saturday', 'Sunday'] else 0,
    'Saison': (date_diffusion.month % 12 // 3 + 1)
  }

  input_data = pd.DataFrame([film_to_predict])
  # Séparer les genres pour appliquer l'encodage MultiLabelBinarizer
  input_data['Genres'] = input_data['Genres'].apply(lambda x: x.split(','))

  # Séparer les nationalités par des slash
  input_data['Nationalité'] = input_data['Nationalité'].apply(lambda x: [i.strip() for i in x.split('/')] if isinstance(x, str) else [])

  # Encoder les nouvelles données
  input_data_encoded = encoder.transform(input_data[['Chaîne', 'Jour']])
  input_data['Vacances scolaires'] = vacances_encoder.transform(input_data['Vacances scolaires'])

  # Normaliser les données numériques
  input_data_scaled = scaler.transform(input_data[['Durée (en min.)', 'IMDB - Note moyenne', 'IMDB - Nombre de votes', 'Année de sortie', 'Année de diffusion', 'Mois', 'Week-end', 'Saison']])

  # Encoder les genres avec MultiLabelBinarizer
  genres_encoded = mlb.transform(input_data['Genres'])

  # Encoder les nationalités
  nationalite_encoded = mlb_nationalite.transform(input_data['Nationalité'])

  # Combiner toutes les features des nouvelles données
  input_data_final = np.hstack([input_data_scaled, input_data_encoded.toarray(), input_data[['Vacances scolaires']].values, genres_encoded, nationalite_encoded])
  st.subheader("Model prediction for your film :")
  prediction = float(model.predict(input_data_final)[0])
  st.write(f"Le film {film_to_predict['TITRE']} diffusé sur {film_to_predict['Chaîne']}, le {date_diffusion} peut espérer une audience de **{round(prediction, 3)} millions** de téléspectateurs.")
