# import random
# import math
# import copy
# import sys


# def fitness(pos):
#     fitnessval = 0.0
#     for i in range(len(pos)):
#         fitnessval += (pos[i]-3)**2
#     return fitnessval

# class Particle:
#     def __init__(self,fitness,dim,minx,maxx,seed):
#         self.rnd = random.Random(seed)
#         self.position = [0.0 for i in range(dim)]
#         self.velocity = [0.0 for i in range(dim)]
#         self.best_part_pos = [0.0 for i in range(dim)]
#         for i in range(dim):
#             self.position[i] = self.rnd.uniform(minx,maxx)
#             self.velocity[i] = self.rnd.uniform(minx,maxx)
#             self.best_part_pos[i] = self.position[i]
#         self.fitness = fitness(self.position)
#         self.best_part_fitness = self.fitness
    
# def pso(fitness,max_iter,n,dim,minx,maxx):
#     w,c1,c2 = 0.8,1.4,1.4
#     rnd = random.Random()
#     swarm = [Particle(fitness,dim,minx,maxx,i) for i in range(n)]
#     best_swarm_pos = [0.0 for i in range(dim)]
#     best_swarm_fitnessval = sys.float_info.max
    
#     for i in range(n):
#         if swarm[i].fitness < best_swarm_fitnessval:
#             best_swarm_fitnessval = swarm[i].fitness
#             best_swarm_pos = copy.copy(swarm[i].position)
    
#     iter = 0
#     while iter<max_iter:
#         if iter%1==0 and iter>1:
#             print("Iter = " + str(iter) + " best fitness = %.3f" % best_swarm_fitnessval)
        
#         for i in range(n):
#             for k in range(dim):
#                 r1 = rnd.random()
#                 r2 = rnd.random()
#                 swarm[i].velocity[k] = w*swarm[i].velocity[k] + c1*r1*(swarm[i].best_part_pos[k]-swarm[i].position[k]) + c2*r2*(best_swarm_pos[k]-swarm[i].position[k])

#                 if swarm[i].velocity[k] > maxx:
#                     swarm[i].velocity[k] = maxx
#                 elif swarm[i].velocity[k] < minx:
#                     swarm[i].velocity[k] = minx
                
#             for k in range(dim):
#                 swarm[i].position[k] = swarm[i].position[k] + swarm[i].velocity[k]
            
#             swarm[i].fitness = fitness(swarm[i].position)
            
#             if swarm[i].fitness < swarm[i].best_part_fitness:
#                 swarm[i].best_part_fitness = swarm[i].fitness
#                 swarm[i].best_part_pos = copy.copy(swarm[i].position)
#                 if swarm[i].fitness < best_swarm_fitnessval:
#                     best_swarm_fitnessval = swarm[i].fitness
#                     best_swarm_pos = copy.copy(swarm[i].position)
                                    
#         iter+=1
#     return best_swarm_pos

# dim = 60
# print("Function has known min = 0.0 at (", end="")
# for i in range(dim-1):
#     print("0, ", end="")
# print("0)")

# n_pop = 500
# max_iter = 1000
# print("Setting num_particles = " + str(n_pop))
# print("Setting max_iter    = " + str(max_iter))
# print("\nStarting PSO algorithm\n")


# best_position = pso(fitness, max_iter, n_pop, dim, -10.0, 10.0)

# print("\nPSO completed\n")
# print("\nBest solution found:")
# print(["%.6f"%best_position[k] for k in range(dim)])
# fitnessVal = fitness(best_position)
# print("fitness of best solution = %.6f" % fitnessVal)
# print("\nEnd particle swarm for sphere function\n")


# import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
import os
import random
import copy
import sys
import datetime


def rmse_train(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def my_loss(y_true, y_pred):
    L1 = np.sum(np.abs(y_true - y_pred))
    L2 = np.sum(np.square(y_true - y_pred))
    mse = K.mean(K.square(y_true - y_pred), axis = -1)
    return L1 + L2 + mse

def predict_point_by_point(model, data):
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size, ))
    print ("predict_size:", predicted.size)
    return predicted

def MAE(pre,true):
    c = 0
    b = abs(np.subtract(pre, true))
    for i in range(len(b)):
        c = c + b[i]
    return c / len(b)

def MAPE(pre, true):
    a=0
    for i in range(len(pre)):
        x = (abs(pre[i] - true[i])/true[i])
        a = a + x
    return a / len(pre)

def RMSE(pre,true):
    c = 0
    b = abs(np.subtract(pre, true))
    b = b*b
    for i in range(len(b)):
        c = c + b[i]
    d = (c / len(b))**0.5
    return d


#Splitting into x and y
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


def run(batch_size,n_steps,units1=50,units2=50,epoch=60):
  attr = 'Lane 1 Flow (Veh/5 Minutes)'
  df = pd.read_csv('pems_output4.csv')
  d1=df.reset_index()['Lane 1 Flow (Veh/5 Minutes)']
  scaler=MinMaxScaler(feature_range=(0,1))
  d1=scaler.fit_transform(np.array(d1).reshape(-1,1))

  # choose a number of time steps
  X, y = split_sequences(d1, n_steps)

  ##splitting dataset into train and test split
  training_size=int(len(d1)*0.80)
  test_size=len(d1)-training_size
  train_data,test_data=X[0:training_size,:],X[training_size:len(d1),:]
  train_l,test_l=y[0:training_size],y[training_size:len(d1)]

  model=Sequential()
  model.add(LSTM(units1,return_sequences=True,input_shape=(n_steps,1)))
  model.add(LSTM(units2,return_sequences=False))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error',optimizer='adam')
  # print(model.summary())
  print("training on : ",units1,units2,epoch,batch_size,n_steps)
  model.fit(train_data,train_l,batch_size=batch_size, epochs=epoch, validation_split=0.15,  verbose=0)
  # print("error before fitting")
  test_predict=model.predict(test_data)
  test_predict=scaler.inverse_transform(test_predict)
  test_l= test_l.reshape(-1, 1)
  test_l=scaler.inverse_transform(test_l)
  loss_metric = {"MAE": MAE(test_l,test_predict),"MAPE": MAPE(test_l,test_predict),"RMSE": RMSE(test_l,test_predict)}
  print("RMSE loss : ",loss_metric["RMSE"][0])
  return loss_metric

def split_sequences(sequences, n_steps):
  X, y = list(), list()
  for i in range(len(sequences)):
    # find the end of this pattern
    end_ix = i + n_steps
    # check if we are beyond the dataset
    if end_ix > len(sequences)-1:
      break
    # gather input and output parts of the pattern
    seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
    X.append(seq_x)
    y.append(seq_y)
  return np.array(X), np.array(y)

def fitness(pos):
    loss_metric = run(batch_size = int(pos[0]),n_steps = int(pos[1]))
    return loss_metric['RMSE']

def check_pos(pos):
  for i in range(len(pos)):
    if pos[i]>max[i]:
      pos[i] = max[i]
    if pos[i]<min[i]:
      pos[i] = min[i]
  return pos

class Particle:
    def __init__(self,fitness,dim,minv,maxv,seed):
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(dim)]
        self.velocity = [0.0 for i in range(dim)]
        self.best_part_pos = [0.0 for i in range(dim)]
        for i in range(dim):
            self.position[i] = self.rnd.uniform(min[i],max[i])
            self.velocity[i] = self.rnd.uniform(minv,maxv)
            self.best_part_pos[i] = self.position[i]
        # print(self.position)
        self.position = check_pos(self.position)
        # print(self.position)
        self.fitness = fitness(self.position)
        self.best_part_fitness = self.fitness

def pso(fitness,max_iter,n,dim,minv,maxv):
    w,c1,c2 = 0.8,1.4,1.4
    rnd = random.Random()
    swarm = [Particle(fitness,dim,minv,maxv,i+12) for i in range(n)]
    best_swarm_pos = [0.0 for i in range(dim)]
    best_swarm_fitnessval = sys.float_info.max
    for i in range(n):
        if swarm[i].fitness < best_swarm_fitnessval:
            best_swarm_fitnessval = swarm[i].fitness
            best_swarm_pos = copy.copy(swarm[i].position) 
  
    iter = 0
    while iter<max_iter:
        print("###########################################################################\n\niter  = ",iter)
        if iter%1==0:
            print("Iter = " + str(iter) + " best fitness = %.3f" % best_swarm_fitnessval)
        for i in range(n):
            for k in range(dim):
                r1 = rnd.random()
                r2 = rnd.random()
                swarm[i].velocity[k] = w*swarm[i].velocity[k] + c1*r1*(swarm[i].best_part_pos[k]-swarm[i].position[k]) + c2*r2*(best_swarm_pos[k]-swarm[i].position[k])

                if swarm[i].velocity[k] > maxv:
                    swarm[i].velocity[k] = maxv
                elif swarm[i].velocity[k] < minv:
                    swarm[i].velocity[k] = minv

                swarm[i].position[k] = swarm[i].position[k] + swarm[i].velocity[k]
            swarm[i].position = check_pos(swarm[i].position)
            swarm[i].fitness = fitness(swarm[i].position)
            
            if swarm[i].fitness < swarm[i].best_part_fitness:
                swarm[i].best_part_fitness = swarm[i].fitness
                swarm[i].best_part_pos = copy.copy(swarm[i].position)
                if swarm[i].fitness < best_swarm_fitnessval:
                    best_swarm_fitnessval = swarm[i].fitness
                    best_swarm_pos = copy.copy(swarm[i].position)                             
        iter+=1
    return best_swarm_pos

dim = 2
n_pop = 5
max_iter = 50
min = [1,2]
max = [64,50]
print("Setting num_particles = " + str(n_pop))
print("Setting max_iter    = " + str(max_iter))
print("\nStarting PSO algorithm\n")
best_position = pso(fitness, max_iter, n_pop, dim,-5.0,5.0)
print("\nPSO completed\n")
print("\nBest solution found:")
print(["%.6f"%best_position[k] for k in range(dim)])
fitnessVal = fitness(best_position)
print("fitness of best solution = %.6f" % fitnessVal)
print("\nEnd particle swarm for sphere function\n")