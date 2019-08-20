import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import linear_model
data = pd.read_csv('homeprices.csv')
# data = pd.read_excel('homePrice.exl')
# area = np.array([2600, 3000, 3200, 3600, 4000])
# price = np.array([550000, 565000, 610000, 680000, 725000])

plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(data.area, data.price, color = 'red', marker ='D')
plt.show()
# as model.fit expect 2D array
new_data = data.drop('price', axis='columns')
print(new_data)

model = linear_model.LinearRegression()
model.fit(new_data, data.price)

# calculte directing by estimating paramaters i.e. slope and intercept
p = model.predict([[5000]])
print(p)
print('slope :', model.coef_)
print('intercept :', model.intercept_)

# predicting for all areas given in a file and saving output data in new file
area_df = pd.read_csv('areas.csv')
t = model.predict(area_df)
print(t)
area_df['prices'] = t
print(area_df)

area_df.to_csv('prediction.csv')
