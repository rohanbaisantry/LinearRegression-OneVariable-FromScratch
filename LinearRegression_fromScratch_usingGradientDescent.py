#LINEAR REGRESSION FOR ONE VARIABLE 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the file
training_data = np.array(pd.read_csv("train.csv", sep=",",header= None) )
testing_data = np.array(pd.read_csv("test.csv", sep=",",header= None))

m=len(training_data) # Number of rows of training data
n=len(testing_data) # number of rows of testing data
lr=0.0001 # learning rate initially
C=[0]*m # array holding the value of the cost function over various weights and bias
b_g=[0]*m # array holding the updated b values
w_g=[0]*m # array holding the updated w values
x_test=[0]*n
y_test=[0]*n
x_train=[0]*m
y_train=[0]*m
for i in range(m):
    x_train[i]=training_data[i,0]
    y_train[i]=training_data[i,1]
for i in range(n):
    x_test[i]=testing_data[i,0]
    y_test[i]=testing_data[i,1]

# functions for gradient descent

def min_cost(b_current, w_current, points, lr):
    co=0
    b_gradient = 0
    w_gradient = 0
    for i in range(m):
        x= points[i,0]
        y= points[i,1]
        b_gradient += -(2/float(m)) * (y - ((w_current * x) + b_current))
        w_gradient += -(2/float(m)) * x * (y - ((w_current * x) + b_current))
    new_b= b_current - (lr * b_gradient)
    new_w= w_current - (lr * w_gradient)
    return [new_b, new_w]

# function to calculate root mean squared error
def errorcal(b,w,points,l):
    points=np.array(points)
    #import pdb
    #pdb.set_trace()
    te=0
    #try:
    for i in range(l):
        x=points[i,0]
        y=points[i,1]
        te+=(y-(w*x+b))**2
    te=(te/float(m))**(0.5)
    return te 

def gradient_descent(points, starting_b, starting_w, lr):
    b= starting_b
    w= starting_w
    #pi=0
    for i in range(m):
        b, w = min_cost(b, w, np.array(points), lr)
        b_g[i]=b
        w_g[i]=w
        C[i]=errorcal(b,w,points,m)
        #pi+=1
        #b_g[i]=b
        #w_g[i]=w
    return [b, w]



# main()

# y = b + w*x
# b = bias  and  w = weight
b_initial=0.01
w_initial=0.02

print (" Initial : b = 0.01, w = 0.02, RMSE = " + str(errorcal(b_initial,w_initial,training_data,m)),"\n\n")

lr= float(input(" What is the learning rate ( alpha ) to be used? ( default is 0.0001, press 0 to set to default.) ) :"))
if lr==0:
    lr=0.0001

b_final, w_final=gradient_descent(training_data,b_initial,w_initial,lr)
print ("\n\n Final : b = " + str(b_final) + ", w = " + str(w_final) + ", RMSE = " + str(errorcal(b_final,w_final,training_data,m)),"\n\n")

#testing 

y_pred=[0]*n # will hold the testing data's predicted values

for i in range(n):
    y_pred[i]=(b_final + w_final*testing_data[i,0])
    
"""
# To compare the actual and predicted values.
for i in range(n):
    print(" actual : " + str(testing_data[i,0]) + " || predicted : " + str(y_pred[i]))
"""
print("\n RMSE whle testing = " + str(errorcal(b_final,w_final,testing_data,n)))

# The points to be plotted in the graph
XG=b_g
YG=w_g
ZG=np.array([C,b_g])
XYZ=np.array([C,XG,yG])

#plotting 3D
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XG, YG, ZG)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
np.savetxt("foo.csv", XYZ, delimiter=',', comments="")

# END
