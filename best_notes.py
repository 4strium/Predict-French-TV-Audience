import pandas as pd

# Chargement du dataset
df = pd.read_csv("database_final.csv")

# Conversion de "IMDB - Note moyenne" en type numérique (si nécessaire)
df["IMDB - Note moyenne"] = pd.to_numeric(df["IMDB - Note moyenne"], errors="coerce")

df["Téléspectateurs (en millions)"] = pd.to_numeric(df["Téléspectateurs (en millions)"], errors="coerce")

# Suppression des valeurs nulles pour éviter les erreurs
df_cleaned = df.dropna(subset=["IMDB - Note moyenne", "Chaîne", "Téléspectateurs (en millions)"])

# Calcul de la moyenne des notes par chaîne
average_ratings = df_cleaned.groupby("Chaîne")["IMDB - Note moyenne"].mean().reset_index()
audience_ratings = df_cleaned.groupby("Chaîne")["Téléspectateurs (en millions)"].mean().reset_index()

# Joindre les deux dataframes sur l'attribut "Chaîne"
average_ratings = pd.merge(average_ratings, audience_ratings, on="Chaîne")

# Renommer les colonnes pour une meilleure lisibilité
average_ratings.columns = ["Chaîne", "Note moyenne", "Téléspectateurs en moyenne (en millions)"]

# Trier par note moyenne dans l'ordre croissant
sorted_ratings = average_ratings.sort_values(by="Note moyenne", ascending=True)

# Limiter les valeurs à deux chiffres après la virgule
sorted_ratings["Note moyenne"] = sorted_ratings["Note moyenne"].round(2)
sorted_ratings["Téléspectateurs en moyenne (en millions)"] = sorted_ratings["Téléspectateurs en moyenne (en millions)"].round(2)

# Afficher le tableau annexe
print(sorted_ratings)

# Sauvegarder le tableau en CSV (facultatif)
sorted_ratings.to_csv("tableau_annexe_chaînes.csv", index=False)
