import random
import math
import csv
import numpy as np
from math import e


learning_rate = .0001
train =[]
test = []
train2= []
test2 = []
corr1=[]
def train_test_split(filename, train=[],test=[], corr=[], split=0.8):
  
   with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        fonts = list(lines)

   pred= []
   for x in range(len(fonts)):
    for y in range(4):
        fonts[x][y] = float(fonts[x][y])

   for x in range(len(fonts)):
       pred.append(fonts[x][3])

   corr.append(pred)
   
   a= np.mean(fonts[x])
   b= np.std(fonts[x])
   value =0
   z = np.delete(fonts, -1, axis=1)
   #print(z)
   for x in range(len(fonts)):
    if value < split*len(fonts):
        train.append(((z[x] -a )/b))
        train2.append(fonts[x][3])
        value+=1
    else:
        test.append(((z[x] -a )/b))
        test2.append(fonts[x][3])

   #print(train)
  
def sigmoid(set):
    return 1/(1+ e**(-set))
   
def weightBias(trainingSet,testingSet,corr):  
    trainings = trainingSet
    testing= testingSet
    #trainings.shape[0]
    #for i in range(len(testing)):
    #print(trainings)
    X_batch= np.split(trainingSet,2000)
    D= X_batch[0].shape[1]
    N= X_batch[1].shape[0]
    M= 16
    K=1
    bias = np.random.randn(M)
    weights=  np.random.randn(D,M)
    
      
    
    

    weights2=  np.random.randn(M,K)
    bias2=  np.random.randn(K)

    Z= []
    Z2= []
    corre = np.array(np.split(corr[0],2500))
    for i in range(len(X_batch)):
        A=  np.dot(X_batch[i], weights) + bias 
        Z= sigmoid(A)
        
        A_2=  np.dot(Z,weights2) + bias2 
        Z2 = sigmoid(A_2)
        #print (weights2.shape)
        getAccuracy(corre[i],Z2)
        cost(Z2, corre[i], corr[0], learning_rate)
        gradientCost(Z2, corre[i])
        backProp(gradientCost(Z2, corre[i]), Z2, Z,X_batch[i])

    
   # print(corre[0])
'''
    for r in range(10000):
          if corr[0][r] == Z2[0]:
            print("C")
    for i in range(10000):
        if corr[0][i] == Z2[i]:
            print("C")

   
'''
   
    

def getAccuracy (testings, predictions):
   # print(testings.shape)
    #print(predictions.shape)  

    #print(testings)
    correct = 0
    for x in range(len(testings)):
        if testings[x] == np.rint(predictions[x]):
            correct+=1
    return (correct/(float(len(testings)))*100)

def cost(predictions, testings, X_batch,learning_rate):
    
    cost_Array = []
    y= testings
    x= predictions
    
    for i in range(len(testings)):
      preds = -(1) * (y[i]*np.log(x[i]) + (1-y[i])*np.log(1-x[i]))
      cost_Array.append(preds)
    
      costs = np.asarray(cost_Array)

    return costs

def gradientCost(predictions, testings):

    y= testings
    x= predictions
    x=x.flatten()
   # print(y.shape)
   # print(x.shape)
    predi = -(1) * (y/x - ((1-y)/(1-x)))
    
    return predi
    
    
def backProp(gradient_Cost,predictions, second_data,X_batch):

  gradient_pred= (1/(1+e**(-second_data)))*(1-1/(1+e**(-second_data)))
  gradient_pred2= (1/(1+e**(-predictions)))*(1-1/(1+e**(-predictions)))
  #print(predictions.shape) *gradient_pred*X_batch
  check = []
  for i in range(len(gradient_pred2)):
      preds = (gradient_Cost[i]*gradient_pred2[i])
      check.append(preds)
    
      ds = np.asarray(check)
  #ds= np.multiply(gradient_Cost,gradient_pred2)
  descent =  np.dot(second_data.T, ds)
  print(ds.shape)

def run():
    trainingSet=[]
    testingSet=[]
    corr= []
    train_test_split('test.csv', trainingSet , testingSet, corr, split=.8)
    corr1= np.array(corr)
    train = np.array(trainingSet)
    test= np.array(testingSet)
    weightBias(train,test,corr1)
    #print(testingSet)

run()