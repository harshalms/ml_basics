import pandas as pd 
from word2number import w2n
import numpy as np 
import math
from sklearn import linear_model

df = pd.read_csv('hiring.csv')
print(df)
df.experience = df.experience.fillna('zero')
print(df)
df.experience = df.experience.apply(w2n.word_to_num)
print(df)
median_test_score = math.floor(df['test_score(out of 10)'].mean())
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_score)
print(df)
reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df['salary($)'])
p = reg.predict([[2, 9, 6]])
print("Salary:", p, '$')