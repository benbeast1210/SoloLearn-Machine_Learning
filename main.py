# Key Libraries
import math
import pandas as pd
import numpy as np

# Project #1: What's In A Column?
def column():
  df = pd.read_csv(filename) 
  arr = np.array(df[column_name]) 
  # print(arr)
  return arr

print(column())

# Project #2: Bob The Builder
def bob():
  n = int(input())
  X = []
  
  for i in range(n):
    X.append([float(x) for x in input().split()])

  y = [int(x) for x in input().split()]

  datapoint = [float(x) for x in input().split()]

  from sklearn.linear_model import LogisticRegression

  model = LogisticRegression()
  model.fit(X,y)

  datapoint = np.array(datapoint).reshape(1,-1)

  print(model.predict(datapoint[[0]])[0])

print(bob())

# Project #3: Welcome to the Matrix
def matrix():
  tp, fp, fn, tn = [int(x) for x in input().split()]

  total = tp + fp + fn + tn
  accuracy = (tp + tn) / total
  precision = tp / (tp + fp)
  recall = tp / (tp + fn) 

  rec = float(recall)
  pre = float(precision)
  numerator = ( 2 * precision * recall)
  f1score = numerator / (precision + recall)

  accuracy = "{:.4g}".format(accuracy)
  precision = "{:.4g}".format(precision)
  recall = "{:.4g}".format(recall)
  f1score = "{:.4g}".format(f1score)

  print(accuracy)
  print(precision)
  print(recall)
  print(f1score)

print(matrix())

# Project #4: Split to Achieve Gain
S = [int(x) for x in input().split()]
A = [int(x) for x in input().split()]
B = [int(x) for x in input().split()]

def split(data):
  c = len(data)
  l = data.count(1)
  return (2 * l / c * (1 - (l / c))) * len(data) / len(S)
  
k = split(S) - split(A) - split(B)
print(round(k, 5))

# Project #5: A Forest of Trees
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def forest():
  randomState = int(input())
  n = int(input())
  rows = []
  for i in range(n):
  	rows.append([float(a) for a in input().split()])
  
  X = np.array(rows)
  y = np.array([int(a) for a in input().split()])
  
  X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=randomState)
  
  rf = RandomForestClassifier(n_estimators=5,random_state=randomState)
  rf.fit(X_train,y_train)
  
  print(rf.predict(X_test))

print(forest())

# Project #6: The Sigmoid Function
def sigmoid():
  w1, w2, b, x1, x2 = [float(x) for x in input().split()]
  output = (w1 * x1) + (w2 * x2) + b
  output = round(1 / (1 + math.exp(output *- 1)), 4)
  print(output)

print(sigmoid())