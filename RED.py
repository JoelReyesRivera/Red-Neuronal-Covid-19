import pandas as pd
import numpy as np

dataTest = pd.read_csv("DatosCovidTest.csv")
dataTrain = pd.read_csv("DatosCovidTrainData.csv")
resultColumn = dataTrain['RESULTADO']
resultColumnTest = dataTest['RESULTADO']
dataTrain = dataTrain.drop('RESULTADO', axis=1)
dataTest = dataTest.drop('RESULTADO', axis=1)

HIDDEN_LAYER_SIZE=10

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(HIDDEN_LAYER_SIZE, activation = 'sigmoid', input_dim = 18))
classifier.add(Dense(units = 6, activation = 'sigmoid'))
classifier.add(Dense(units = 3, activation = 'sigmoid'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
classifier.fit(dataTrain, resultColumn, batch_size = 1, epochs = 10)
metrics = classifier.evaluate(dataTest, resultColumnTest, batch_size=128)
print("%s: %.2f%%" % (classifier.metrics_names[1],metrics[1]*100))
results = classifier.predict(dataTest)
print(results)
classifier.summary()