import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sys

simplefilter("ignore", category=ConvergenceWarning)

df = pd.read_csv('/home/runner/kaggle/titanic/processed_titanic_data.csv')

features_to_use = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0']

columns = ['Survived'] + features_to_use
df = df[columns]
df = df.head(100)

def check_classifier_value(knn, input_df, row_index):
    independent_variables = input_df[[column for column in df.columns if column != 'Survived']]
    dependent_variable = input_df['Survived']

    values = independent_variables.iloc[[row_index]].to_numpy().tolist()[0]

    training_df = independent_variables.drop([row_index])
    X = training_df.reset_index(drop=True).to_numpy().tolist()
    testing_df = dependent_variable.drop([row_index])
    Y = testing_df.reset_index(drop=True).to_numpy().tolist()

    predicted_value = knn.fit(X, Y).predict([values])

    if predicted_value == input_df['Survived'].iloc[[row_index]].to_numpy().tolist()[0]:
        return True

    return False


def get_accuracy(knn, input_df):
    correct_classifications = 0

    for row_index in range(len(input_df.to_numpy().tolist())):
        if check_classifier_value(knn, input_df, row_index):
            correct_classifications += 1

    return correct_classifications / len(input_df.to_numpy().tolist())


accuracies = []
#simple_scaling_accuracies = []
#min_max_accuracies = []
#z_score_accuracies = []
k_values = [1, 3, 5, 10, 15, 20, 30, 40, 50, 75]


#simple_scaling_df = df.copy()
#min_max_df = df.copy()
#z_score_df = df.copy()

#for feature in features_to_use:
    #simple_scaling_df[feature] = simple_scaling_df[feature] / simple_scaling_df[feature].max()

    #min_max_df[feature] = (min_max_df[feature] - min_max_df[feature].min()) / (min_max_df[feature].max() - min_max_df[feature].min())

    #z_score_df[feature] = (z_score_df[feature] - z_score_df[feature].mean()) / (z_score_df[feature].std())

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors = k)

    accuracies.append(get_accuracy(knn, df))

    #simple_scaling_accuracies.append(get_accuracy(knn, simple_scaling_df))

    #min_max_accuracies.append(get_accuracy(knn, min_max_df))

    #z_score_accuracies.append(get_accuracy(knn, z_score_df))

plt.style.use("bmh")
plt.plot(k_values, accuracies, label='no normalizing')
#plt.plot(k_values, simple_scaling_accuracies, label='simple scaling')
#plt.plot(k_values, min_max_accuracies, label='min-max scaling')
#plt.plot(k_values, z_score_accuracies, label='z-scoring')
plt.xlabel("k")
plt.ylabel("accuracy")
plt.title("Leave One Out vs K")
plt.legend(loc='best')
plt.savefig('knn.png')
