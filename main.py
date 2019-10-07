import random
import math
import csv
import numpy as np
from math import e
import matplotlib
import matplotlib.pyplot as plt


learning_rate = .0001
train =[]
test = []
train2= []
test2 = []
corr1=[]
W = np.zeros((4,16))
B = np.zeros(16)
W2 = np.zeros((16,1))
B2= np.zeros(1)
boole = False
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

    B = np.random.randn(M)
    W=  np.random.randn(D,M)
        
    W2 =  np.random.randn(M,K)
    B2=  np.random.randn(K)
    
    Z= []
    Z2= []
    adW= []
    adB= []
    adW2= []
    adB2= []

    corre = np.array(np.split(corr[0],2500))
    for i in range(len(X_batch)):
        if boole == False:
            A=  np.dot(X_batch[i], W) + B 
            Z= sigmoid(A)
            
            A_2=  np.dot(Z,W2) + B2 
            Z2 = sigmoid(A_2)
        else:
            Arr= adjust(W, B, W2, B2, adW, adB, adW2, adB2)
            A=  np.dot(X_batch[i], Arr[0]) + Arr[1]
            Z= sigmoid(A)
    
            A_2=  np.dot(Z, Arr[2]) + Arr[3]
            Z2 = sigmoid(A_2)

        getAccuracy(corre[i],Z2)
        G= cost(Z2, corre[i], corr[0])
        gradientCost(Z2, corre[i])
        v=backProp(gradientCost(Z2, corre[i]), Z2, Z,X_batch[i])
        adW= v[0]
        adB= v[1]
        adW2= v[2]
        adW= v[3]
        

    
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

def cost(predictions, testings, X_batch):
    
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
    
    
def backProp(gradient_Cost,predictions, second_data,first_data):

  gradient_pred= (1/(1+e**(-second_data)))*(1-1/(1+e**(-second_data)))
  gradient_pred2= (1/(1+e**(-predictions)))*(1-1/(1+e**(-predictions)))

  check = []
  check_0 =[]
  for i in range(len(gradient_pred2)):
      preds = (gradient_Cost[i]*gradient_pred2[i])
      preds_0 = (gradient_Cost[i]*gradient_pred2[i]*gradient_pred[i])

      check.append(preds)
      check_0.append(preds_0)

  ds = np.asarray(check)
  ds_0 = np.asarray(check_0)
  
  weights2 =  np.dot(second_data.T, ds)
  bias2 =  ds

  

  weights= np.dot(first_data.T, ds_0)
  bias = ds_0
  boole =True
  return [weights,bias,weights2, bias2]


def adjust (W, B, W2, B2, adW, adB, adW2, adB2):

    new_W= W- learning_rate*adW   
    new_B= B- learning_rate*adB   
    new_W2= W2- learning_rate*adW2   
    new_B2= B2- learning_rate*adB2   
    return [new_W,new_B,new_W2, new_B2]
    '''
    A=  np.dot(batch, new_W) + new_B 
    Z= sigmoid(A)

    A_2=  np.dot(Z,new_W2) + new_B2 
    Z2 = sigmoid(A_2)
    '''
def plot(costs):
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    fig, graph = plt.subplots()

    graph.plot(t, s)

    graph.set(xlabel='iterations', ylabel='loss',
       title='Ok')
    graph.grid()

   fig.savefig("test.png")
   plt.show()  

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