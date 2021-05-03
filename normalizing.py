import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sys


df = pd.read_csv('/home/runner/kaggle/book-data.csv')

features_to_use = [f for f in df.columns if f != 'book type']

columns = ['book type'] + features_to_use
df = df[columns]


def convert_book_type_to_number(book_type):
    if "children" in book_type:
        return 0

    if "adult" in book_type:
        return 1


df['book type'] = df['book type'].apply(lambda entry: convert_book_type_to_number(entry))


def check_classifier_value(knn, input_df, row_index):
    independent_variables = input_df[[column for column in df.columns if column != 'book type']]
    dependent_variable = input_df['book type']

    values = independent_variables.iloc[[row_index]].to_numpy().tolist()[0]

    training_df = independent_variables.drop([row_index])
    X = training_df.reset_index(drop=True).to_numpy().tolist()
    testing_df = dependent_variable.drop([row_index])
    Y = testing_df.reset_index(drop=True).to_numpy().tolist()

    predicted_value = knn.fit(X, Y).predict([values])

    if predicted_value == input_df['book type'].iloc[[row_index]].to_numpy().tolist()[0]:
        return True

    return False


def get_accuracy(knn, input_df):
    correct_classifications = 0

    for row_index in range(len(input_df.to_numpy().tolist())):
        if check_classifier_value(knn, input_df, row_index):
            correct_classifications += 1

    return correct_classifications / len(input_df.to_numpy().tolist())


accuracies = []
simple_scaling_accuracies = []
min_max_accuracies = []
z_score_accuracies = []
k_values = [n for n in range(1, 100, 2)]

simple_scaling_df = df.copy()
min_max_df = df.copy()
z_score_df = df.copy()

for f in features_to_use:
    simple_scaling_df[f] = simple_scaling_df[f] / simple_scaling_df[f].max()

    min_max_df[f] = (min_max_df[f] - min_max_df[f].min()) / (min_max_df[f].max() - min_max_df[f].min())

    z_score_df[f] = (z_score_df[f] - z_score_df[f].mean()) / (z_score_df[f].std())

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors = k)

    accuracies.append(get_accuracy(knn, df))

    simple_scaling_accuracies.append(get_accuracy(knn, simple_scaling_df))

    min_max_accuracies.append(get_accuracy(knn, min_max_df))

    z_score_accuracies.append(get_accuracy(knn, z_score_df))

plt.style.use("bmh")
plt.plot(k_values, accuracies, label='unscaled')
plt.plot(k_values, simple_scaling_accuracies, label='simple scaling')
plt.plot(k_values, min_max_accuracies, label='min-max')
plt.plot(k_values, z_score_accuracies, label='z-scoring')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Leave One Out Accuracy for various Normalization")
plt.legend(loc='best')
plt.savefig('normalizing.png')
