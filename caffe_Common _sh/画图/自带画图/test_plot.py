import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
file=open('out.log.test')
information = file.readlines()
Iters_train = []
Iters_test = []

TrainingLoss = []
LearningRate = []
for inf in information[1:len(information) - 1]:
    tmp = inf.split()
    Iters_train.append(tmp[0])
    TrainingLoss.append(tmp[2])
    #LearningRate.append(tmp[3])

Iters_train = np.array(Iters_train)
TrainingLoss = np.array(TrainingLoss)
#LearningRate = np.array(LearningRate)
plt.figure()
plt.plot(Iters_train,TrainingLoss)
#####################
file=open('out.log.train')
information = file.readlines()
Iters_train = []
Iters_test = []

TrainingLoss = []
LearningRate = []
for inf in information[1:len(information) - 1]:
    tmp = inf.split()
    Iters_train.append(tmp[0])
    TrainingLoss.append(tmp[2])
    #LearningRate.append(tmp[3])

Iters_train = np.array(Iters_train)
TrainingLoss = np.array(TrainingLoss)
#LearningRate = np.array(LearningRate)

plt.plot(Iters_train,TrainingLoss)
plt.savefig('test.png')
