import pandas

df = pandas.read_csv('title.akas.tsv', sep='\t')

df = df[df['region'] == 'FR']

df.to_csv('akas_fr.csv', index=False)
