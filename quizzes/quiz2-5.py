import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

df = pd.read_csv('/home/runner/kaggle/StudentsPerformance.csv')
df = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'math score', 'reading score', 'writing score']]

print("Math scores for last  rows:\n", df['math score'][-3:], "\n")

print("Mean Score:", df['math score'].mean(), "\n")


test_prep_complete = (df['test preparation course'] == 'completed')
no_test_prep = (df['test preparation course'] == 'none')

print("Math scores for people who did test prep course:", df['math score'][test_prep_complete].mean())

print("Math score for people who didn't do test prep course:", df['math score'][no_test_prep].mean(), "\n")

print("# of parental education fields:", len(df['parental level of education'].unique()), "\n")

dummy_columns = ['test preparation course', 'parental level of education']
keep_columns = ['math score'] + dummy_columns
df = df[keep_columns]

for column in dummy_columns:
    unique_values = df[column].unique()

    for value in unique_values:
        dummy_column_name = '{}={}'.format(column, value)
        df[dummy_column_name] = df[column].apply(lambda x: int (x==value))

    del df[column]


data = np.array(df)
train_arr = data[:-3, :]
test_arr = data[-3:, :]

X_train = train_arr[:, 1:]
X_test = test_arr[:, 1:]

y_train = train_arr[:, 0]
y_test = test_arr[:, 0]

regressor = LinearRegression()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print("Actual Values:", y_test)
print("Predictions:", predictions)
