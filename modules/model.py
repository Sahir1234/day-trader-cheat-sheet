
import matplotlib.pyplot as plt
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


class Model (object):

    def __init__(self, num_inputs):
        
        self.model = Sequential()
        
        # The Input Layer :
        self.model.add(Dense(128, kernel_initializer='normal',input_dim = num_inputs, activation='relu'))

        # The Hidden Layers :
        self.model.add(Dense(256, kernel_initializer='normal',activation='relu'))
        self.model.add(Dense(256, kernel_initializer='normal',activation='relu'))
        self.model.add(Dense(256, kernel_initializer='normal',activation='relu'))

        # The Output Layer :
        self.model.add(Dense(1, kernel_initializer='normal',activation='linear'))
        
        self.model.compile(loss='mean_absolute_error', optimizer='adam')

        print(self.model.summary())
    

    def train_test(self, x, y):
        
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 52)
        
        loss_history = self.model.fit(X_train, Y_train, epochs=100, batch_size=128, verbose=1)
        
        y_pred = self.model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
        
        print('*****')
        print('ROOT MEAN SQUARED ERROR: ' + str(rmse))
        print('*****')
        
        return loss_history
    
    def plot_loss(self, loss_history):
    
        plt.plot(loss_history.history['loss'])
        plt.title('Model Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
