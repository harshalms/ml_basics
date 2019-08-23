import numpy as np
import pandas as pd
from sklearn import linear_model
# import math

df = pd.read_csv('homeprices1.csv')
print(df,'\n')

df.bedrooms = df.bedrooms.fillna(df.bedrooms.median()) # .fillna fills values at NaN
print(df,'\n')

model = linear_model.LinearRegression()
model.fit(df.drop('price', axis='columns'),df.price) # model.fit trains the model with training set

p = model.predict([[3000, 3, 40]])  # expect 2D matrix as a input for prediction
print('House price for (3000 sq. ft., 3 bedrooms, age:40) :', p,'\n')

print(model.coef_,'\n') # all parameters
print(model.intercept_) # theta0