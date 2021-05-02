import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

df = pd.read_csv('/home/runner/kaggle/data-scientist-hiring.csv')

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

less_than_10 = 0

for size in df['company_size']:
    size = str(size)
    max_size = int(size[3:])

    if max_size < 10:
        less_than_10 += 1

    if size == NaN:
        less_than_10 += 0

print(less_than_10)