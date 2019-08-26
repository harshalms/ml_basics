import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('carprices.csv')
# print(df)

dummies = pd.get_dummies(df['Car Model'])
# print(dummies)
# df = np.hstack((df, dummies)) # This makes a matrix and not a table
df_dummies = pd.concat([df, dummies], axis = 'columns')
# print(df_dummies)

# df_dummies.drop('Car Model', axis = 'columns', inplace = True)
# df_dummies.drop('Mercedez Benz C class', axis = 'columns', inplace = True)
# x = df_dummies.drop('Sell Price($)', axis = 'columns')

x = df_dummies.drop(['Car Model','Mercedez Benz C class', 'Sell Price($)'], axis = 'columns')
y = df_dummies['Sell Price($)']
# print(df_dummies)
# print(x)
# print(y)
# Training our model with training examples
model = linear_model.LinearRegression()
model.fit(x, y)
# Prediction
score = model.score(x, y)
print('score :', score)
p = model.predict([[45000, 4, 0, 0]])
print(p)
# Getting input from a csv file and generating new file with prediction
t = pd.read_csv('predict_carprices.csv')
new_dummies = pd.get_dummies(t['Car Model'])
t_dummies = pd.concat([t, new_dummies], axis = 'columns')
t_new = t_dummies.drop(['Car Model', 'Mercedez Benz C class'], axis = 'columns')
t_result = model.predict(t_new)
# print(t_result)
t['Price($)'] = t_result
t.to_csv('car_prediction.csv')
