import pandas

df_basics = pandas.read_csv('data-init/title.basics.tsv', sep='\t')
df_complete = pandas.read_csv('database_complete.csv')

df_complete['Année de sortie'] = df_complete['IMDB ID'].map(df_basics.set_index('tconst')['startYear'])
df_complete['Genres'] = df_complete['IMDB ID'].map(df_basics.set_index('tconst')['genres'])
df_complete['Durée (en min.)'] = df_complete['IMDB ID'].map(df_basics.set_index('tconst')['runtimeMinutes'])

df_complete.to_csv('database_final.csv', index=False)
