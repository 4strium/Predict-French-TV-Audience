import pandas

df_data = pandas.read_csv('clean-process/database_fine.csv')
df_ratings = pandas.read_csv('data-init/title.ratings.tsv', sep='\t')

df_data['IMDB - Note moyenne'] = df_data['IMDB ID'].map(df_ratings.set_index('tconst')['averageRating'])
df_data['IMDB - Nombre de votes'] = df_data['IMDB ID'].map(df_ratings.set_index('tconst')['numVotes'])

missing_imdb_id_count = df_data['IMDB - Note moyenne'].isna().sum()
print(f"Nombre de valeurs vides pour la colonne 'IMDB - Note moyenne': {missing_imdb_id_count}")

print(df_data.head())
df_data.to_csv('database_complete.csv', index=False)
