import pandas
import datetime

df = pandas.read_csv('database.csv',sep=";")

# Prétraitement :
df.iloc[:, -2:] = df.iloc[:, -2:].replace(',', '.', regex=True)
df['Téléspectateurs (en millions)'] = df['Téléspectateurs (en millions)'].astype(float)
df["Part d'audience (en %)"] = df["Part d'audience (en %)"].astype(float)
df['Date de diffusion'] = pandas.to_datetime(df['Date de diffusion'], format='%d/%m/%Y')
df['Jour'] = df['Date de diffusion'].dt.day_name()
df['Nationalité'] = df['Nationalité'].str.replace('DE', 'Allemagne', regex=False)
df['Nationalité'] = df['Nationalité'].str.replace('CH', 'Suisse', regex=False)
df['Nationalité'] = df['Nationalité'].str.replace('CA', 'Canada', regex=False)

vacances_scolaires = [
    # Année 2023
    (datetime.datetime(2023, 2, 4), datetime.datetime(2023, 2, 20)),  # Vacances d'hiver
    (datetime.datetime(2023, 4, 8), datetime.datetime(2023, 4, 24)),  # Vacances de printemps
    (datetime.datetime(2023, 7, 8), datetime.datetime(2023, 9, 4)),   # Vacances d'été
    (datetime.datetime(2023, 10, 21), datetime.datetime(2023, 11, 6)), # Vacances de la Toussaint
    (datetime.datetime(2023, 12, 23), datetime.datetime(2024, 1, 8)),  # Vacances de Noël

    # Année 2022
    (datetime.datetime(2022, 2, 5), datetime.datetime(2022, 2, 21)),
    (datetime.datetime(2022, 4, 9), datetime.datetime(2022, 4, 25)),
    (datetime.datetime(2022, 7, 9), datetime.datetime(2022, 9, 4)),
    (datetime.datetime(2022, 10, 22), datetime.datetime(2022, 11, 7)),
    (datetime.datetime(2022, 12, 17), datetime.datetime(2023, 1, 2)),

    # Année 2021
    (datetime.datetime(2021, 2, 6), datetime.datetime(2021, 2, 22)),
    (datetime.datetime(2021, 4, 10), datetime.datetime(2021, 5, 3)),
    (datetime.datetime(2021, 7, 10), datetime.datetime(2021, 9, 1)),
    (datetime.datetime(2021, 10, 16), datetime.datetime(2021, 11, 2)),
    (datetime.datetime(2021, 12, 18), datetime.datetime(2022, 1, 3)),

    # Année 2020
    (datetime.datetime(2020, 2, 15), datetime.datetime(2020, 3, 2)),
    (datetime.datetime(2020, 4, 4), datetime.datetime(2020, 4, 19)),
    (datetime.datetime(2020, 7, 4), datetime.datetime(2020, 9, 1)),
    (datetime.datetime(2020, 10, 17), datetime.datetime(2020, 11, 2)),
    (datetime.datetime(2020, 12, 19), datetime.datetime(2021, 1, 4)),

    # Année 2019
    (datetime.datetime(2019, 2, 9), datetime.datetime(2019, 2, 25)),
    (datetime.datetime(2019, 4, 13), datetime.datetime(2019, 4, 29)),
    (datetime.datetime(2019, 7, 6), datetime.datetime(2019, 9, 2)),
    (datetime.datetime(2019, 10, 19), datetime.datetime(2019, 11, 4)),
    (datetime.datetime(2019, 12, 21), datetime.datetime(2020, 1, 6)),

    # Année 2018
    (datetime.datetime(2018, 2, 10), datetime.datetime(2018, 2, 26)),
    (datetime.datetime(2018, 4, 7), datetime.datetime(2018, 4, 23)),
    (datetime.datetime(2018, 7, 7), datetime.datetime(2018, 9, 3)),
    (datetime.datetime(2018, 10, 20), datetime.datetime(2018, 11, 5)),
    (datetime.datetime(2018, 12, 22), datetime.datetime(2019, 1, 7)),

    # Année 2017
    (datetime.datetime(2017, 2, 11), datetime.datetime(2017, 2, 27)),
    (datetime.datetime(2017, 4, 8), datetime.datetime(2017, 4, 24)),
    (datetime.datetime(2017, 7, 8), datetime.datetime(2017, 9, 4)),
    (datetime.datetime(2017, 10, 21), datetime.datetime(2017, 11, 6)),
    (datetime.datetime(2017, 12, 23), datetime.datetime(2018, 1, 8)),

    # Année 2016
    (datetime.datetime(2016, 2, 6), datetime.datetime(2016, 2, 22)),
    (datetime.datetime(2016, 4, 9), datetime.datetime(2016, 4, 25)),
    (datetime.datetime(2016, 7, 9), datetime.datetime(2016, 9, 4)),
    (datetime.datetime(2016, 10, 22), datetime.datetime(2016, 11, 7)),
    (datetime.datetime(2016, 12, 17), datetime.datetime(2017, 1, 2)),

    # Année 2015
    (datetime.datetime(2015, 2, 14), datetime.datetime(2015, 3, 2)),
    (datetime.datetime(2015, 4, 4), datetime.datetime(2015, 4, 20)),
    (datetime.datetime(2015, 7, 4), datetime.datetime(2015, 9, 1)),
    (datetime.datetime(2015, 10, 17), datetime.datetime(2015, 11, 2)),
    (datetime.datetime(2015, 12, 19), datetime.datetime(2016, 1, 4)),

    # Année 2014
    (datetime.datetime(2014, 2, 15), datetime.datetime(2014, 3, 3)),
    (datetime.datetime(2014, 4, 12), datetime.datetime(2014, 4, 28)),
    (datetime.datetime(2014, 7, 5), datetime.datetime(2014, 9, 1)),
    (datetime.datetime(2014, 10, 18), datetime.datetime(2014, 11, 3)),
    (datetime.datetime(2014, 12, 20), datetime.datetime(2015, 1, 5)),

    # Année 2013
    (datetime.datetime(2013, 2, 9), datetime.datetime(2013, 2, 25)),
    (datetime.datetime(2013, 4, 13), datetime.datetime(2013, 4, 29)),
    (datetime.datetime(2013, 7, 6), datetime.datetime(2013, 9, 2)),
    (datetime.datetime(2013, 10, 19), datetime.datetime(2013, 11, 4)),
    (datetime.datetime(2013, 12, 21), datetime.datetime(2014, 1, 6)),

    # Année 2012
    (datetime.datetime(2012, 2, 11), datetime.datetime(2012, 2, 27)),
    (datetime.datetime(2012, 4, 7), datetime.datetime(2012, 4, 23)),
    (datetime.datetime(2012, 7, 7), datetime.datetime(2012, 9, 3)),
    (datetime.datetime(2012, 10, 20), datetime.datetime(2012, 11, 5)),
    (datetime.datetime(2012, 12, 22), datetime.datetime(2013, 1, 7)),

    # Année 2011
    (datetime.datetime(2011, 2, 12), datetime.datetime(2011, 2, 28)),
    (datetime.datetime(2011, 4, 9), datetime.datetime(2011, 4, 25)),
    (datetime.datetime(2011, 7, 9), datetime.datetime(2011, 9, 5)),
    (datetime.datetime(2011, 10, 22), datetime.datetime(2011, 11, 7)),
    (datetime.datetime(2011, 12, 17), datetime.datetime(2012, 1, 2)),

    # Année 2010
    (datetime.datetime(2010, 2, 6), datetime.datetime(2010, 2, 22)),
    (datetime.datetime(2010, 4, 10), datetime.datetime(2010, 4, 26)),
    (datetime.datetime(2010, 7, 10), datetime.datetime(2010, 9, 6)),
    (datetime.datetime(2010, 10, 23), datetime.datetime(2010, 11, 8)),
    (datetime.datetime(2010, 12, 18), datetime.datetime(2011, 1, 3)),

    # Année 2009
    (datetime.datetime(2009, 2, 7), datetime.datetime(2009, 2, 23)),
    (datetime.datetime(2009, 4, 11), datetime.datetime(2009, 4, 27)),
    (datetime.datetime(2009, 7, 4), datetime.datetime(2009, 9, 1)),
    (datetime.datetime(2009, 10, 17), datetime.datetime(2009, 11, 2)),
    (datetime.datetime(2009, 12, 19), datetime.datetime(2010, 1, 4)),

    # Année 2008
    (datetime.datetime(2008, 2, 9), datetime.datetime(2008, 2, 25)),
    (datetime.datetime(2008, 4, 12), datetime.datetime(2008, 4, 28)),
    (datetime.datetime(2008, 7, 5), datetime.datetime(2008, 9, 1)),
    (datetime.datetime(2008, 10, 18), datetime.datetime(2008, 11, 3)),
    (datetime.datetime(2008, 12, 20), datetime.datetime(2009, 1, 5)),

    # Année 2007
    (datetime.datetime(2007, 2, 10), datetime.datetime(2007, 2, 26)),
    (datetime.datetime(2007, 4, 7), datetime.datetime(2007, 4, 23)),
    (datetime.datetime(2007, 7, 7), datetime.datetime(2007, 9, 3)),
    (datetime.datetime(2007, 10, 20), datetime.datetime(2007, 11, 5)),
    (datetime.datetime(2007, 12, 22), datetime.datetime(2008, 1, 7)),

    # Année 2006
    (datetime.datetime(2006, 2, 11), datetime.datetime(2006, 2, 27)),
    (datetime.datetime(2006, 4, 8), datetime.datetime(2006, 4, 24)),
    (datetime.datetime(2006, 7, 8), datetime.datetime(2006, 9, 4)),
    (datetime.datetime(2006, 10, 21), datetime.datetime(2006, 11, 6)),
    (datetime.datetime(2006, 12, 23), datetime.datetime(2007, 1, 8)),

    # Année 2005
    (datetime.datetime(2005, 2, 12), datetime.datetime(2005, 2, 28)),
    (datetime.datetime(2005, 4, 9), datetime.datetime(2005, 4, 25)),
    (datetime.datetime(2005, 7, 9), datetime.datetime(2005, 9, 1)),
    (datetime.datetime(2005, 10, 22), datetime.datetime(2005, 11, 7)),
    (datetime.datetime(2005, 12, 17), datetime.datetime(2006, 1, 2)),

    # Année 2004
    (datetime.datetime(2004, 2, 14), datetime.datetime(2004, 3, 1)),
    (datetime.datetime(2004, 4, 10), datetime.datetime(2004, 4, 26)),
    (datetime.datetime(2004, 7, 10), datetime.datetime(2004, 9, 6)),
    (datetime.datetime(2004, 10, 23), datetime.datetime(2004, 11, 8)),
    (datetime.datetime(2004, 12, 18), datetime.datetime(2005, 1, 3)),

    # Année 2003
    (datetime.datetime(2003, 2, 8), datetime.datetime(2003, 2, 24)),
    (datetime.datetime(2003, 4, 12), datetime.datetime(2003, 4, 28)),
    (datetime.datetime(2003, 7, 5), datetime.datetime(2003, 9, 1)),
    (datetime.datetime(2003, 10, 18), datetime.datetime(2003, 11, 3)),
    (datetime.datetime(2003, 12, 20), datetime.datetime(2004, 1, 5)),

    # Année 2002
    (datetime.datetime(2002, 2, 9), datetime.datetime(2002, 2, 25)),
    (datetime.datetime(2002, 4, 13), datetime.datetime(2002, 4, 29)),
    (datetime.datetime(2002, 7, 6), datetime.datetime(2002, 9, 2)),
    (datetime.datetime(2002, 10, 19), datetime.datetime(2002, 11, 4)),
    (datetime.datetime(2002, 12, 21), datetime.datetime(2003, 1, 6)),

    # Année 2001
    (datetime.datetime(2001, 2, 10), datetime.datetime(2001, 2, 26)),
    (datetime.datetime(2001, 4, 7), datetime.datetime(2001, 4, 23)),
    (datetime.datetime(2001, 7, 7), datetime.datetime(2001, 9, 3)),
    (datetime.datetime(2001, 10, 20), datetime.datetime(2001, 11, 5)),
    (datetime.datetime(2001, 12, 22), datetime.datetime(2002, 1, 7)),

    # Année 2000
    (datetime.datetime(2000, 2, 12), datetime.datetime(2000, 2, 28)),
    (datetime.datetime(2000, 4, 8), datetime.datetime(2000, 4, 24)),
    (datetime.datetime(2000, 7, 8), datetime.datetime(2000, 9, 4)),
    (datetime.datetime(2000, 10, 21), datetime.datetime(2000, 11, 6)),
    (datetime.datetime(2000, 12, 23), datetime.datetime(2001, 1, 8)),

    # Année 1999
    (datetime.datetime(1999, 2, 13), datetime.datetime(1999, 2, 28)),
    (datetime.datetime(1999, 4, 10), datetime.datetime(1999, 4, 26)),
    (datetime.datetime(1999, 7, 10), datetime.datetime(1999, 9, 1)),
    (datetime.datetime(1999, 10, 23), datetime.datetime(1999, 11, 8)),
    (datetime.datetime(1999, 12, 18), datetime.datetime(2000, 1, 3)),
]

def est_vacances(date):
    for debut, fin in vacances_scolaires:
        if debut <= date <= fin:
            return 'oui'
    return 'non'

df['Vacances scolaires'] = df['Date de diffusion'].apply(est_vacances)

df.to_csv('database_fine.csv', index=False)
