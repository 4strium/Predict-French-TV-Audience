import streamlit as st
import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from vacances_scolaires_france import SchoolHolidayDates
import requests

st.title("French TV Audience Prediction")

# ======================================================
# 1. LOAD DATA (CACHE OK : PAS DE LISTES)
# ======================================================
@st.cache_data
def load_data():
    return pd.read_csv("database.csv")

data = load_data()

# ======================================================
# 2. PREPROCESSING (HORS CACHE)
# ======================================================
data['Date de diffusion'] = pd.to_datetime(data['Date de diffusion'])
data['Mois'] = data['Date de diffusion'].dt.month
data['Année de diffusion'] = data['Date de diffusion'].dt.year
data['Week-end'] = data['Jour'].isin(['Saturday', 'Sunday']).astype(int)
data['Saison'] = data['Mois'] % 12 // 3 + 1
data['Nationalité'] = data['Nationalité'].str.upper()

# Genres / Nationalités → listes (HORS CACHE)
data['Genres'] = data['Genres'].str.split(',')
data['Nationalité'] = data['Nationalité'].apply(
    lambda x: [i.strip() for i in x.split('/')] if isinstance(x, str) else []
)

# Encoders multilabel (globaux, pas dans le cache)
mlb_genres = MultiLabelBinarizer()
genres_df = pd.DataFrame(
    mlb_genres.fit_transform(data['Genres']),
    columns=mlb_genres.classes_
)

mlb_nat = MultiLabelBinarizer()
national_df = pd.DataFrame(
    mlb_nat.fit_transform(data['Nationalité']),
    columns=mlb_nat.classes_
)

features = [
    'Chaîne', 'Jour', 'Durée (en min.)',
    'IMDB - Note moyenne', 'IMDB - Nombre de votes',
    'Année de sortie', 'Année de diffusion',
    'Mois', 'Week-end', 'Saison', 'Vacances scolaires'
]
target = 'Téléspectateurs (en millions)'

# ======================================================
# 3. TRAINING (CACHE RESOURCE – OK)
# ======================================================
@st.cache_resource
def train_model(data, genres_df, national_df):
    encoder = OneHotEncoder(handle_unknown="ignore")
    scaler = StandardScaler()
    vacances_encoder = LabelEncoder()

    X = data[features].copy()
    y = data[target]

    X_cat = encoder.fit_transform(X[['Chaîne', 'Jour']])
    X_num = scaler.fit_transform(X[
        ['Durée (en min.)', 'IMDB - Note moyenne', 'IMDB - Nombre de votes',
         'Année de sortie', 'Année de diffusion', 'Mois', 'Week-end', 'Saison']
    ])

    X['Vacances scolaires'] = vacances_encoder.fit_transform(X['Vacances scolaires'])

    X_final = np.hstack([
        X_num,
        X_cat.toarray(),
        X[['Vacances scolaires']].values,
        genres_df.values,
        national_df.values
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    model = xgboost.XGBRegressor(
        max_depth=3,
        subsample=0.91,
        tree_method='hist',
        seed=42,
        n_estimators=30,
        learning_rate=0.32
    )
    model.fit(X_train, y_train)

    return model, encoder, scaler, vacances_encoder, X_test, y_test


if "model" not in st.session_state:
    with st.spinner("🧠 Training model..."):
        (st.session_state.model,
         st.session_state.encoder,
         st.session_state.scaler,
         st.session_state.vacances_encoder,
         st.session_state.X_test,
         st.session_state.y_test) = train_model(data, genres_df, national_df)

# ======================================================
# 4. METRICS
# ======================================================
model = st.session_state.model
y_pred = model.predict(st.session_state.X_test)

st.subheader("Model infos :")
st.write("RMSE :", np.sqrt(mean_squared_error(st.session_state.y_test, y_pred)))
st.write("R² :", r2_score(st.session_state.y_test, y_pred))

# ======================================================
# 5. PREDICTION FORM
# ======================================================
st.subheader("Predict your film :")

with st.form("prediction_form"):
    imdb_id = st.text_input("IMDB Film ID (ttxxxxx)")
    channel = st.selectbox("Channel", [
        "TF1", "France 2", "France 3", "France 4", "France 5",
        "M6", "Arte", "C8", "W9", "TMC", "TFX",
        "TF1 Séries Films", "6ter", "Gulli",
        "Canal +", "C Star", "NRJ12", "Chérie 25"
    ])
    date_diffusion = st.date_input("Broadcast date", format="DD/MM/YYYY")
    submitted = st.form_submit_button("Predict !")

# ======================================================
# 6. INFERENCE
# ======================================================
if submitted and imdb_id:
    all_data_json = requests.get(
        f"https://api.imdbapi.dev/titles/{imdb_id}"
    ).json()

    input_data = pd.DataFrame([{
        'Chaîne': channel,
        'Jour': pd.Timestamp(date_diffusion).day_name(),
        'Durée (en min.)': all_data_json['runtimeSeconds'] // 60,
        'IMDB - Note moyenne': float(all_data_json['rating']['aggregateRating']),
        'IMDB - Nombre de votes': int(all_data_json['rating']['voteCount']),
        'Année de sortie': int(all_data_json['startYear']),
        'Année de diffusion': date_diffusion.year,
        'Mois': date_diffusion.month,
        'Week-end': int(pd.Timestamp(date_diffusion).day_name() in ['Saturday', 'Sunday']),
        'Saison': date_diffusion.month % 12 // 3 + 1,
        'Vacances scolaires': 'oui' if SchoolHolidayDates().is_holiday(date_diffusion) else 'non'
    }])

    X_cat = st.session_state.encoder.transform(input_data[['Chaîne', 'Jour']])
    X_num = st.session_state.scaler.transform(input_data[
        ['Durée (en min.)', 'IMDB - Note moyenne', 'IMDB - Nombre de votes',
         'Année de sortie', 'Année de diffusion', 'Mois', 'Week-end', 'Saison']
    ])
    input_data['Vacances scolaires'] = st.session_state.vacances_encoder.transform(
        input_data['Vacances scolaires']
    )

    genres_encoded = mlb_genres.transform([all_data_json['genres']])
    nat_encoded = mlb_nat.transform([[c.upper() for c in all_data_json['originCountries']]])

    X_final = np.hstack([
        X_num,
        X_cat.toarray(),
        input_data[['Vacances scolaires']].values,
        genres_encoded,
        nat_encoded
    ])

    prediction = model.predict(X_final)[0]

    st.subheader("Prediction")
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
      st.image(all_data_json['primaryImage']['url'], width=200)
    st.success(f"Audience estimée : **{prediction:.3f} millions**")
