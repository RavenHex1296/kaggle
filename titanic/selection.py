import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import sys

simplefilter("ignore", category=ConvergenceWarning)

df = pd.read_csv('/home/runner/kaggle/titanic/dataset_of_knowns.csv')

wanted_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
df = df[wanted_columns]

#Sex
def convert_sex_to_int(sex):
   if sex == 'male':
       return 0

   elif sex == 'female':
       return 1

df['Sex'] = df['Sex'].apply(convert_sex_to_int)

#Age
age_nan = df['Age'].apply(lambda entry: np.isnan(entry))
age_not_nan = df['Age'].apply(lambda entry: not np.isnan(entry))

mean_age = df['Age'][age_not_nan].mean()
df.loc[age_nan, ['Age']] = mean_age

#SipSp
def indicator_greater_than_zero(x):
   if x > 0:
       return 1

   else:
       return 0

df['SibSp>0'] = df['SibSp'].apply(indicator_greater_than_zero)

#Parch
df['Parch>0'] = df['Parch'].apply(indicator_greater_than_zero)
del df['Parch']

#CabinType
df['Cabin'] = df['Cabin'].fillna('None')

def cabin_type(cabin):
   if cabin != 'None':
       return cabin[0]

   else:
       return cabin


df['CabinType'] = df['Cabin'].apply(cabin_type)

for cabin_type in df['CabinType'].unique():
   dummy_variable_name = 'CabinType={}'.format(cabin_type)
   dummy_variable_values = df['CabinType'].apply(lambda entry: int(entry==cabin_type))
   df[dummy_variable_name] = dummy_variable_values

del df['CabinType']

#Embarked
df['Embarked'] = df['Embarked'].fillna('None')

for embark in df['Embarked'].unique():
   dummy_variable_name = 'Embarked={}'.format(embark)
   dummy_variable_values = df['Embarked'].apply(lambda entry: int(entry==embark))
   df[dummy_variable_name] = dummy_variable_values

del df['Embarked']


features_to_use = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0', 'Parch>0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']

interactions = {}
used_features = []

for feature in features_to_use:
   if 'SibSp' not in feature and 'Embarked=' not in feature and 'CabinType=' not in feature:
       used_features.append(feature)
       interactions[feature] = [non_redundant_feature for non_redundant_feature in features_to_use if non_redundant_feature not in used_features]

   if 'SibSp' in feature:
       used_features.append('SibSp')
       used_features.append('SibSp>0')
       interactions[feature] = [non_redundant_feature for non_redundant_feature in features_to_use if non_redundant_feature not in used_features]

   if 'Embarked=' in feature:
       used_features.append('Embarked=C')
       used_features.append('Embarked=None')
       used_features.append('Embarked=Q')
       used_features.append('Embarked=S')
       interactions[feature] = [non_redundant_feature for non_redundant_feature in features_to_use if non_redundant_feature not in used_features]

   if 'CabinType=' in feature:
       used_features.append('CabinType=A')
       used_features.append('CabinType=B')
       used_features.append('CabinType=C')
       used_features.append('CabinType=D')
       used_features.append('CabinType=E')
       used_features.append('CabinType=F')
       used_features.append('CabinType=G')
       used_features.append('CabinType=None')
       used_features.append('CabinType=T')
       interactions[feature] = [non_redundant_feature for non_redundant_feature in features_to_use if non_redundant_feature not in used_features]


for feature, non_redundant_features in interactions.items():
   for non_redundant_feature in non_redundant_features:
       interaction_term = '{} * {}'.format(feature, non_redundant_feature)
       df[interaction_term] = df[feature] * df[non_redundant_feature]
       features_to_use += [interaction_term]


columns = ['Survived'] + features_to_use
df = df[columns]

training_df = df[:500]
testing_df = df[500:]


def convert_regressor_output_to_survival_value(n):
   if n < 0.5:
       return 0

   else:
       return 1


def get_accuracy(predictions, actual):
   correct_predictions = 0
   incorrect_predictions = 0

   for n in range(len(predictions)):
       if predictions[n] == actual[n]:
           correct_predictions += 1

       else:
           incorrect_predictions += 1

   return correct_predictions / (correct_predictions + incorrect_predictions)


def get_set_accuracy(df, features, acc_type):
    if len(features) == 0:
        return 0

    columns = ['Survived'] + features
    training_array = np.array(training_df[columns])
    testing_array = np.array(testing_df[columns])

    y_train = training_array[:, 0]
    y_test = testing_array[:, 0]

    X_train = training_array[:, 1:]
    X_test = testing_array[:, 1:]

    regressor = LogisticRegression(max_iter=100, random_state=0)
    regressor.fit(X_train, y_train)

    if acc_type == "Train":
        y_train_predictions = regressor.predict(X_train)
        y_train_predictions = [convert_regressor_output_to_survival_value(prediction) for prediction in y_train_predictions]
        return get_accuracy(y_train_predictions, y_train)

    if acc_type == "Test":
        y_test_predictions = regressor.predict(X_test)
        y_test_predictions = [convert_regressor_output_to_survival_value(prediction) for prediction in y_test_predictions]
        return get_accuracy(y_test_predictions, y_test)


def next_feature(df, features, wanted_features):
   best_accuracy = get_set_accuracy(df, wanted_features, "Test")
   best_feature = None

   for feature in features:
       if feature not in wanted_features:
           test_features = wanted_features + [feature]
           accuracy = get_set_accuracy(df, test_features, "Test")

           if accuracy > best_accuracy:
               best_accuracy = accuracy
               best_feature = feature

   return best_feature

'''
wanted_features = [next_feature(df, features_to_use, [])]
training_accuracy = get_set_accuracy(df, ['Sex'], "Train")
testing_accuracy = get_set_accuracy(df, ['Sex'], "Test")

print(wanted_features)
print("training:", training_accuracy)
print("testing:", testing_accuracy, "\n")

next_possible_feature = next_feature(df, features_to_use, wanted_features)

while next_possible_feature != None:
    wanted_features.append(next_possible_feature)
    training_accuracy =  get_set_accuracy(df, wanted_features, "Train")
    testing_accuracy = get_set_accuracy(df, wanted_features, "Test")
    print(wanted_features)
    print("training:", training_accuracy)
    print("testing:", testing_accuracy, "\n")
    next_possible_feature = next_feature(df, features_to_use, wanted_features)
'''


all_features = [feature for feature in df.columns if feature != 'Survived']
baseline_features = [feature for feature in all_features]
baseline_testing_accuracy = get_set_accuracy(df, baseline_features, "Test")
print("Training:", get_set_accuracy(df, all_features, "Train"))
print("Testing:", get_set_accuracy(df, all_features, "Test"), "\n")
removed_indices = []

for index, feature in enumerate(all_features):
    baseline_features.remove(feature)
    training_accuracy = get_set_accuracy(df, baseline_features, "Train")
    testing_accurary = get_set_accuracy(df, baseline_features, "Test")
    print("Candidate to remove: {} (index {})".format(feature, index))
    print("Training:", training_accuracy)
    print("Testing:", testing_accurary)

    if testing_accurary < baseline_testing_accuracy:
        baseline_features.insert(index, feature)
        print("Kept")


    else:
        baseline_testing_accuracy = testing_accurary
        removed_indices.append(index)
        print("Removed")

    print("Baseline testing accuracy:", baseline_testing_accuracy)
    print("Removed indices:", removed_indices)
    print("")

print(baseline_testing_accuracy)