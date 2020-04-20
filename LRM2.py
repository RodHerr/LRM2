#!/usr/bin/env python
# coding: utf-8

# Libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Dataset
# 
# Fishes market, modified measures from different fish types, obtained from:https://www.kaggle.com/aungpyaeap/fish-market.
# 
# This just displays the data obtained from the csv file

data=pd.read_csv("Fish/Fishes.csv")
print(data.info())
print(data.head())


# Shuffle data
# 
# In order to get some random train and test data, we shuffle the data and then take some for train and we leave the others to test. Thish shows that the data is no longer in order.

data=data.sample(len(data))
print(data)


# Train and Test
# 
# Here we separate the train and test independent and dependent variables

X=["Length1","Length2","Length3","Height","Width"];Y="Weight"
xtrain=data[X][0:1000]; ytrain=(data[Y][0:1000])**.3
# # print(xtrain); print(ytrain)
xtest=data[X][1000:1100]; ytest=(data[Y][1000:1100])**.3
# # print(xtest); print(ytest)


# Plot relation
# 
# This plot shows how all independen variables are behaving, knowing tht a linear regression model can be implemented

plt.figure(1);plt.xlabel("Independent vars");
plt.ylabel("Weight");
plt.scatter(xtrain["Length1"],ytrain,label="Length1")
plt.scatter(xtrain["Length2"],ytrain,label="Length2")
plt.scatter(xtrain["Length3"],ytrain,label="Length3")
plt.scatter(xtrain["Height"],ytrain,label="Height")
plt.scatter(xtrain["Width"],ytrain,label="Width")
plt.legend()


# Hypothesis
# 
# Function implemented for the predictions with the linear model.

def hyp(weights,x):
    hyp=0
    for i in range(len(weights)):
        hyp+=(weights[i]*x[i])
    return hyp


# Errors
# 
# Shows the actual improvement for the error mean and stores it for further plotting

evol=[]
def  error(weights,x,y):
    erracum=0
    for i in range(len(x)):
        p=hyp(weights,x[i])
#         print("p(x)=%f y(x)=%f" %(p,y[i]))
        erracum+=(p-y[i])**2
    errmean=erracum/len(x)
    evol.append(errmean)
    print("em=%f" %(errmean))


# Gradient Descent
# 
# Give the actual weights for the model and according to the cost function returns new updated values


def gd(weights,x,y,a):
    tmp=list(weights)
    for j in range(len(weights)):
        acum=0;
        for i in range(len(x)):
            err=hyp(weights,x[i])-y[i]
            acum+=err*x[i][j]
        tmp[j]=weights[j]-a*(1/len(x))*acum
    return tmp


# Main (Training Model)
# 
# At an alpha "a", run the whole code "gens" times, showing improvement and then displaying it into a graph.


evol=[]
epoch=0
a=0.0006
gen=5000

weights=[0,0,0,0,0,0]
ylist=ytrain.values.tolist()
xlist=xtrain.values.tolist()
# print(xlist)
for i in range(len(xlist)):
    xlist[i]=[1]+xlist[i]

while True:
    old=list(weights)
    weights=gd(weights,xlist,ylist,a)
    error(weights,xlist,ylist)
    epoch+=1
    if(old==weights or epoch==gen):
        print ("Weights:")
        print (weights)
        break

plt.plot(evol)


# Test
# 
# Evaluate with the test data, show graph relationa and error mean

x = np.linspace(0,len(ytest),len(ytest))
plt.scatter(x,ytest)
p=(hyp(weights,[1,xtest["Length1"],xtest["Length2"],xtest["Length3"],xtest["Height"],xtest["Width"]]))
plt.scatter(x,p)

testem=0
p=p.values.tolist()
y=ytest.values.tolist()
for i in range(len(x)):
    testem+=(p[i]-y[i])**2
print("Test error mean=%f"%(testem/len(x)))


# User queries
# 
# User iputs in order to get a result, dataset headset is display for info.

print(data)

x0=float(input("Enter Length1:"))
x1=float(input("Enter Length2:"))
x2=float(input("Enter Length3:"))
x3=float(input("Enter Height:"))
x4=float(input("Enter Width:"))
pred=(hyp(weights,[1,x0,x1,x2,x3,x4]))**3.33333333333
print(pred)



