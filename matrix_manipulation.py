import numpy as np

# matrix initialization
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
print('x =',x)
print('y =', y)
# add() is used to add matrices
print('Addition of x and y:', np.add(x, y))
print('x + y =', x+y)
# subtract() is used to subtract matrices
print('Subtraction of x and y:', np.subtract(x, y))
print('x - y =', x-y)
# divide() is used to divide matrices
print('Matrix division', np.divide(x,y))
print('x/y =', x/y)
print('Multiplication of two matrices:', np.multiply(x, y))
print('Element wise multiplication, x*y =', x*y)
print('The product of two matrices :', np.dot(x, y))
print('square root is:', np.sqrt(x))
print ("The summation of elements : ")
print (np.sum(y))
print ("The column wise summation  : ")
print (np.sum(y,axis=0))
print ("The row wise summation: ")
print (np.sum(y,axis=1))
# using "T" to transpose the matrix
print('Matrix transposition: ')
print(x.T)
