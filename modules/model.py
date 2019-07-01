

# Import necessary packages for data manipulation and visualization,
# model construction and metrics for evaluation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import LambdaCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Defining some model parameters to improve training and results
OPTIMIZER = 'adam'
BATCH_SIZE = 30
EPOCHS = 150
INITIALIZER='normal'
LOSS = 'mean_squared_error'
ACTIVATION='relu'

# This class serves as the
class Model (object):

    def __init__(self, num_inputs):
        
        self.model = Sequential()
        
        
        # The Input Layer :
        self.model.add(Dense(16, kernel_initializer=INITIALIZER, input_dim = num_inputs, activation=ACTIVATION))

        
        # The Hidden Layers :
        self.model.add(Dense(32, kernel_initializer=INITIALIZER, activation=ACTIVATION))
        
        self.model.add(Dense(64, kernel_initializer=INITIALIZER, activation=ACTIVATION))
        
        self.model.add(Dense(128, kernel_initializer=INITIALIZER, activation=ACTIVATION))
        
        self.model.add(Dense(64, kernel_initializer=INITIALIZER, activation=ACTIVATION))
        
        self.model.add(Dense(32, kernel_initializer=INITIALIZER, activation=ACTIVATION))
        
        
        # The Output Layer :
        self.model.add(Dense(1, kernel_initializer=INITIALIZER, activation='linear'))

        
        self.model.compile(loss=LOSS, optimizer=OPTIMIZER)

        print(self.model.summary())
    

    def train(self, x, y, X_pred):
        
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.1, random_state = 52)
        
        prediction_history = []
        
        callback =  LambdaCallback(on_epoch_end=lambda epoch, logs: prediction_history.append(self.model.predict(X_pred).item(0)))
        
        loss_history = self.model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[callback])
        
        y_pred = self.model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
        
        print('*****')
        print('ROOT MEAN SQUARED ERROR: ' + str(rmse))
        print('*****')
        
        return loss_history, prediction_history, rmse
    
    def plot_loss(self, loss_history):
    
        plt.plot(loss_history.history['loss'])
        plt.title('Model Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    def plot_predictions(self, prediction_history):
        
        plt.plot(prediction_history)
        plt.title('Model Prediction History')
        plt.xlabel('Epoch')
        plt.ylabel('Predictions')
        plt.show()
    

    def predict(self, X_pred):
        
        Y_pred = self.model.predict(X_pred, verbose=1)
        
        closing_price = Y_pred.item(0)
        
        print('************************')
        print('')
        print('CLOSING PRICE PREDICTION: $' + str(closing_price))
        print('')
        print('************************')

        return closing_price
