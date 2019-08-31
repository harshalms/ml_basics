import numpy as np
import pandas as pd
from word2number import w2n
import math

df = pd.read_csv('hiring.csv')
# print(df)
df.experience = df.experience.fillna('zero')
df.experience = df.experience.apply(w2n.word_to_num)
val = math.floor(df['test_score(out of 10)'].mean())
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(val)
# print(df)
# x = df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']]
x = df.drop('salary($)', axis='columns')
y = df['salary($)']
print('y :',y)
# print(x)
rows, cols = x.shape
z = np.ones([rows, 1], dtype = int)
x = np.hstack((z, x))
print(x)

# gradient descent implementation
iterations = 1
alpha = 0.01
m = rows 
theta = np.zeros([cols+1, 1])
for i in range(iterations):
    yp = np.dot(x, theta)
    print('yp :',yp,'\n')
    k = yp-y
    print('k :', k)
    cost = 1/(2*m)*sum(((yp-y))**2)
    print('cost :',cost,'\n')
    grad = (1/m)*np.dot(x.T, (yp-y))
    theta -= alpha*grad
print(theta)
# y = np.dot(x,theta)
# print(y)
# # print(x, y)