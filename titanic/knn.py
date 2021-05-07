import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sys
import time

start_time = time.time()

df = pd.read_csv('/home/runner/kaggle/titanic/processed_titanic_data.csv')

features_to_use = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp']

columns = ['Survived'] + features_to_use
df = df[columns]
df = df.head(100)

num_observations = len(df)

def check_classifier_value(knn, X, y, row_index_to_observe):
    X_observed = X[row_index_to_observe, :]
    X_with_observation_removed = np.delete(X, row_index_to_observe, 0)

    y_observed = y[row_index_to_observe]
    y_with_observation_removed = np.delete(y, row_index_to_observe, 0)

    knn.fit(X_with_observation_removed, y_with_observation_removed)
    predicted_value = knn.predict([X_observed])

    if predicted_value == y_observed:
        return True

    return False


def get_accuracy(knn, X, y):
    correct_classifications = 0

    for row_index in range(num_observations):
        if check_classifier_value(knn, X, y, row_index):
            correct_classifications += 1

    return correct_classifications / num_observations


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


arr = np.array(df)
X = arr[:, 1:]
y = arr[:, 0]

arr1 = np.array(simple_scaling_df)
X_simple_scaling = arr1[:, 1:]

arr2 = np.array(min_max_df)
X_min_max = arr2[:, 1:]

arr3 = np.array(z_score_df)
X_z_score = arr3[:, 1:]


for k in k_values:
    knn = KNeighborsClassifier(n_neighbors = k)

    accuracies.append(get_accuracy(knn, X, y))

    simple_scaling_accuracies.append(get_accuracy(knn, X_simple_scaling, y))

    min_max_accuracies.append(get_accuracy(knn, X_min_max, y))

    z_score_accuracies.append(get_accuracy(knn, X_z_score, y))

end_time = time.time()
print("Time taken:", end_time - start_time)


plt.style.use("bmh")
plt.plot(k_values, accuracies, label='no normalizing')
plt.plot(k_values, simple_scaling_accuracies, label='simple scaling')
plt.plot(k_values, min_max_accuracies, label='min-max scaling')
plt.plot(k_values, z_score_accuracies, label='z-scoring')
plt.xlabel("k")
plt.ylabel("accuracy")
plt.title("Leave One Out vs K")
plt.legend(loc='best')
plt.savefig('titanic/normalizing_titanic_data.png')
