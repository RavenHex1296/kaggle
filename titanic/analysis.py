import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import sys

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
df['Cabin']= df['Cabin'].fillna('None')

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

training_array = np.array(training_df)
testing_array = np.array(testing_df)

y_train = training_array[:,0]
y_test = testing_array[:,0]

X_train = training_array[:,1:]
X_test = testing_array[:,1:]

#regressor = LinearRegression()
regressor = LogisticRegression(max_iter=10000)
regressor.fit(X_train, y_train)

coefficients = {}
feature_columns = training_df.columns[1:]
#feature_coefficients = regressor.coef_
feature_coefficients = regressor.coef_[0]

for n in range(len(feature_columns)):
    column = feature_columns[n]
    coefficient = feature_coefficients[n]
    coefficients[column] = coefficient

y_test_predictions = regressor.predict(X_test)
y_train_predictions = regressor.predict(X_train)

def convert_regressor_output_to_survival_value(n):
    if n < 0.5:
        return 0

    else:
        return 1


y_test_predictions = [convert_regressor_output_to_survival_value(n) for n in y_test_predictions]
y_train_predictions = [convert_regressor_output_to_survival_value(n) for n in y_train_predictions]


def get_accuracy(predictions, actual):
    correct_predictions = 0
    incorrect_predictions = 0

    for n in range(len(predictions)):
        if predictions[n] == actual[n]:
            correct_predictions += 1

        else:
            incorrect_predictions += 1
    
    return correct_predictions / (correct_predictions + incorrect_predictions)


print('\n')
#print("\n", "features:", features_to_use, "\n")
print("training accuracy:", get_accuracy(y_train_predictions, y_train))
print("testing accuracy:", get_accuracy(y_test_predictions, y_test), "\n")

#coefficients['constant'] = regressor.intercept_
#print({k: round(v, 4) for k, v in coefficients.items()})
coefficients['constant'] = regressor.intercept_[0]
#print({k: round(v, 4) for k, v in coefficients.items()})
