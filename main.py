import random
import math
import csv
import numpy as np


learning_rate = .0001
training =[]

def train_test_split(filename, train=[],test=[], split=0.8):
  
   with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        fonts = list(lines)

   for x in range(len(fonts)-1):
    for y in range(4):
        fonts[x][y] = float(fonts[x][y])
          
   a= np.mean(fonts[x])
   b= np.std(fonts[x])
   for x in range(len(fonts)-1):
    if random.random() < split:
        train.append(((fonts[x] -a )/b))
    else:
        test.append(((fonts[x] -a )/b))
   
def weightBias(trainingSet):  
    trainingSet = training
    
def update(initial, learning_rate):
    return 0

def run():
    trainingSet=[]
    testingSet=[]
    train_test_split('test.csv', trainingSet , testingSet, split=.8)
    training = trainingSet
    print(testingSet)

run()