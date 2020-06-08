import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

dataTrain = pd.read_csv("CSV/DataTrain.csv", header=None)
dataTrainRes = pd.read_csv("CSV/DataTrainRes.csv", header=None)
dataTest = pd.read_csv("CSV/DataTest.csv", header=None)
dataTestRes = pd.read_csv("CSV/DataTestRes.csv", header=None)


classifier = Sequential() # Initializa la red neuronal

classifier.add(Dense(units = 10, activation = 'sigmoid', input_dim = 18))
classifier.add(Dense(units = 6, activation = 'sigmoid'))
classifier.add(Dense(units = 3, activation = 'sigmoid'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'sgd', loss = 'mean_squared_error')
classifier.fit(dataTrain, dataTrainRes, batch_size = 1, epochs = 100)

Res_pred = classifier.predict(dataTest)
Res_pred = [ 1 if y>=0.5 else 0 for y in Res_pred ]
print(Res_pred)

total = 0
correct = 0
wrong = 0
for i in range(len(Res_pred)):
  total=total+1
  if(dataTestRes.at[i,0] == Res_pred[i]):
    correct=correct+1
  else:
    wrong=wrong+1

print("Total de datos: " + str(total))
print("Correctos: " + str(correct))
print("Incorrectos: " + str(wrong))