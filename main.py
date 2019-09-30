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
   for x in range(len(fonts)-2):
    if value < split*len(fonts):
        train.append(((fonts[x] -a )/b))
        train2.append(fonts[x][3])
        value+=1
    else:
        test.append(((fonts[x] -a )/b))
        test2.append(fonts[x][3])


  
def sigmoid(set):
    return 1/(1+ e**(-set))
   
def weightBias(trainingSet,testingSet,corr):  
    trainings = trainingSet
    testing= testingSet
    #trainings.shape[0]
    #for i in range(len(testing)):
   # print(testing)
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
        print(getAccuracy(corre[i],Z2))
        #print(X_batch[i])
        
    
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

def update(predictions, testings, X_batch,learning_rate):
    
    cost_Array = []
    N = float(len(testings))
    for i in range(0, len(testings)):
        y= testings[i]
        x= predictions[i]
        pred_gradient = -(1) * (y/x - ((1-y)/(1-x)))
        cost_Array.append(pred_gradient)
    
    cost_Array[: -1]
    
    

    return 0

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