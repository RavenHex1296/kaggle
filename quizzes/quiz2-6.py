import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

df = pd.read_csv('/home/runner/kaggle/quizzes/data-scientist-hiring.csv')

#print("Mean number of training hours:", df['training_hours'].mean(),  "\n")

looking_for_job = (df['target'] == 1.)
need_job = 0

for n in looking_for_job:
    if n:
        need_job += 1

#print("People looking for a job:", need_job / len(df['target']), "\n")


#df = df.groupby(['city']).count()
#most_id = df['enrollee_id'].idxmax()
#print("City with most students:", most_id)
#print("Number of students", df['enrollee_id'][most_id])




most_id = 0

for city in df['city'].unique():
    city_id =  int(city[5:])

    if city_id > most_id:
        most_id = city_id

#print(most_id)


company_size_counts = df['company_size'].value_counts()

#print(company_size_counts['<10'])


total = company_size_counts['<10'] + \
        company_size_counts['10/49'] + \
        company_size_counts['50-99']

#print(total)