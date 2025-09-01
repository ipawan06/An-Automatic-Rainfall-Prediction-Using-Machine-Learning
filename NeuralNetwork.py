import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
from math import sqrt
from sklearn.model_selection import train_test_split
import math
from keras.models import Sequential
from keras.layers import Dense, Activation ,Dropout , Flatten , Conv1D ,MaxPooling1D
from keras.layers.recurrent import LSTM
from keras import losses
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from timeit import default_timer as timer


def build_model(input):
    model = Sequential()
    model.add(Dense(128,input_shape=(input[1],input[0])))
    model.add(Conv1D(filters = 24, kernel_size= 1,padding='valid', activation='relu', kernel_initializer="uniform"))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(Conv1D(filters = 48,kernel_size = 1,padding='valid', activation='relu', kernel_initializer="uniform"))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(LSTM(40,return_sequences=True))
    model.add(LSTM(32,return_sequences=False))
    model.add(Dense(32, activation="relu", kernel_initializer="uniform"))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss='mse',optimizer='adam',metrics=['mae'])
    return model


def process(path,sname,nos):
        df = pd.read_csv(path)
        df=df[df['SUBDIVISION'] == sname]
        df.drop(df.columns[[0,1,15,16,17,18]], axis=1, inplace=True)
        print(df.head())
        print(df.tail)
        #print(df.tail())
        forecast_col='ANNUAL'
        forecast_out = 10
        print(forecast_out)
                
        df['label'] = df[forecast_col].shift(-forecast_out)
        X = df.drop(['label'],1)
        #X = preprocessing.scale(X)
        X_lately = df[-forecast_out:]
        X = X[:-forecast_out]

        df.dropna(inplace = True)
        y = df['label']
        
        print(X)
        print(y)

        df1=pd.concat([X, y],axis=1)
        print(df1)
        data = df1.to_numpy()
        
        print(data)


        result = []
        sequence_length = 6
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])


        result = np.array(result)
        
        row = round(0.8 * result.shape[0])
        
        #creating training data
        train = result[:int(row), :]
        
        x_train = train[:, :-1]
        y_train = train[:, -1][:,-1]
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1][:,-1]
        
        amount_of_features = len(df.columns)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features)) 

        print("X_train", x_train.shape)
        print("y_train", y_train.shape)
        print("X_test", x_test.shape)
        print("y_test", y_test.shape)
        
        
    
        model = build_model([15,5,1])
        #Summary of the Model
        print(model.summary())

        start = timer()
        history = model.fit(x_train,y_train,batch_size=128,epochs=25,validation_split=0.2,verbose=2)
        end = timer()
        print(end - start)


        data = X_lately.to_numpy()
        
        
        result = []
        sequence_length = int(nos)
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])


        result = np.array(result)

        result = np.reshape(result, (result.shape[0], result.shape[1], amount_of_features)) 
        p = model.predict(result)
        print(p)
        preds = X_lately[forecast_col].values[:-5] * (p[0][0] + 1)

        print(preds)

        x = np.arange(5)
        plt.plot(x, preds,label='CNNLSTM')
        for a,b in zip(x, preds):
            plt.text(a, b, str(b))
        plt.legend()
        plt.title('CNNLSTM')
        plt.xlabel("Day (s)")
        plt.ylabel("Predicted Value")
        plt.savefig('results/CNNLSTMForecast.png')
        plt.pause(5)
        plt.show(block=False)
        plt.close()
#process("data.csv","ANDAMAN & NICOBAR ISLANDS")
