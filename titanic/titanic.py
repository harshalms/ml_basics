import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('train.csv')
# print(train_df.head())
# print(train_df.describe())
y = train_df.pop('Survived')
# print(train_df.head())
numeric_variables = list(train_df.dtypes[train_df.dtypes != 'object'].index)
# print(train_df[numeric_variables].tail())
train_df['Age'].fillna(train_df['Age'].mean(), inplace = True)
# print(train_df.tail())
# print(train_df[numeric_variables].tail())

# Logistic Regression Model
# model = linear_model.LogisticRegression()
# model.fit(train_df[numeric_variables], y)
# score = model.score(train_df[numeric_variables], y)
# print('Accuracy Score :',score)

# Random Forest Classifier
model1 = RandomForestClassifier(n_estimators = 100)
model1.fit(train_df[numeric_variables], y)
score1 = model1.score(train_df[numeric_variables], y)
print('Accuracy Score1 :',score1)

test_df = pd.read_csv('test.csv')
test_df['Age'].fillna(test_df['Age'].mean(), inplace = True)
test_df = test_df[numeric_variables].fillna(test_df.mean()).copy()

y_pred = model1.predict(test_df[numeric_variables])
# print(y_pred)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived' : y_pred
})
submission.to_csv('titanic_test_result.csv', index=False)