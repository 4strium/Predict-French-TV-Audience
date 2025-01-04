import pandas
import unidecode
import re

df_data = pandas.read_csv('database_fine.csv')
df_title = pandas.read_csv('akas_fr.csv')
df_ratings = pandas.read_csv('title.ratings.tsv', sep='\t')

df_data['TITRE'] = df_data['TITRE'].apply(lambda x: unidecode.unidecode(x).upper())
df_title['title'] = df_title['title'].apply(lambda x: unidecode.unidecode(x).upper())

df_title = df_title.merge(df_ratings, left_on='titleId', right_on='tconst', how='left')
df_title = df_title.sort_values(by='numVotes', ascending=False).drop_duplicates(subset='title', keep='first')
df_title = df_title.drop(columns=['averageRating', 'numVotes', 'tconst'])

df_data['IMDB ID'] = df_data['TITRE'].map(df_title.set_index('title')['titleId'])
def arabic_to_roman(num):
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
        ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
        ]
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num

def replace_arabic_with_roman(title):
    match = re.search(r'(\d+)$', title)
    if match:
        arabic_num = int(match.group(1))
        roman_num = arabic_to_roman(arabic_num)
        return title[:match.start()] + roman_num
    return title

df_data['TITRE'] = df_data.apply(
    lambda row: replace_arabic_with_roman(row['TITRE']) if pandas.isna(row['IMDB ID']) else row['TITRE'],
    axis=1
)

df_data['IMDB ID'] = df_data['TITRE'].map(df_title.set_index('title')['titleId'])

df_data['IMDB - Note moyenne'] = df_data['IMDB ID'].map(df_ratings.set_index('tconst')['averageRating'])
df_data['IMDB - Nombre de votes'] = df_data['IMDB ID'].map(df_ratings.set_index('tconst')['numVotes'])

missing_imdb_id_count = df_data['IMDB ID'].isna().sum()
print(f"Nombre de valeurs vides pour la colonne 'IMDB ID': {missing_imdb_id_count}")

print(df_data.head())
df_data.to_csv('databse_complete.csv', index=False)
