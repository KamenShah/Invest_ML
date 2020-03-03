from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN
import pickle

batch_size = 32
num_classes = 2
epochs = 20


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



# the data, split between train and test sets
def getData(filename, rebalance):
    
    df = pd.read_csv(filename)
    df = df.replace('None', 0)
    df = df.loc[(df!='Stock').all(1)]
    df = df.dropna(axis=1,how='all')
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df = df[(df != 0).all(1)]


    df = df.drop('Debt Equity Ratio', 1)
    df = df.drop('Dividend Yield', 1)
    df = df.drop('Dividend Payout Ratio', 1)
    df = df.drop('Inventory Turnover', 1)
    df = df.drop('Receivables Growth', 1)
    df = df.drop('Debt Growth', 1)
    df = df.drop('R&D Growth', 1)
    df = df.sample(frac=1).reset_index(drop=True)
    #print(df.astype(bool).sum(axis=0))

    df['Beat Estimate'] = df['Beat Estimate'].map({'-1': 0, '1':1})


    trainSize = round(0.80 * len(df))
    validSize = round(0.90 * len(df))
    train = df[:trainSize]
    valid = df[trainSize:validSize]
    test = df[validSize:]

    if(rebalance):
        x, y = SMOTE().fit_resample(train.loc[:, 'EPS Estimate':'Asset Growth'], train.loc[:, 'Beat Estimate'])

        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        train = pd.concat([y,x], axis= 1)


    train_Y = train.loc[:, 'Beat Estimate'].values.astype(float)
    train_X = train.loc[:, 'EPS Estimate':'Asset Growth'].values.astype(float)
    valid_Y = valid.loc[:, 'Beat Estimate'].values.astype(float)
    valid_X = valid.loc[:, 'EPS Estimate':'Asset Growth'].values.astype(float)
    test_Y = test.loc[:, 'Beat Estimate'].values.astype(float)
    test_X = test.loc[:, 'EPS Estimate':'Asset Growth'].values.astype(float)

    print(train_X.shape)
    print(train_X.shape[1], 'features')
    print(train_X.shape[0], 'train samples')
    print(valid_X.shape[0], 'validation samples')
    print(test_X.shape[0], 'test samples')

    print(str(np.sum(train_Y == 0)) + ' class 0 in train')
    print(str(np.sum(train_Y == 1)) + ' class 1 in train')

    train_Y = keras.utils.to_categorical(train_Y, num_classes)
    valid_Y = keras.utils.to_categorical(valid_Y, num_classes)
    test_Y = keras.utils.to_categorical(test_Y, num_classes)

    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y


def trainModel(train_X, train_Y, valid_X, valid_Y):
    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(19,)))
    model.add(Dropout(0.2))
    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['acc',f1_m,precision_m, recall_m])

    history = model.fit(train_X, train_Y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(valid_X, valid_Y))
    return model, history



def testModel(model, test_X, test_Y):
    score = model.evaluate(test_X, test_Y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def plotMetrics(history):
    # Plot the loss
    fig, (ax,ax2,ax3) = plt.subplots(3, 1, figsize=(10,6))
    ax.plot(np.sqrt(history.history['loss']), 'r', label='train')
    ax.plot(np.sqrt(history.history['val_loss']), 'b' ,label='val')
    ax.set_xlabel(r'Epoch')
    ax.set_ylabel(r'Loss')
    ax.legend()


    # Plot the accuracy
    ax2.plot(np.sqrt(history.history['acc']), 'r', label='train')
    ax2.plot(np.sqrt(history.history['val_acc']), 'b' ,label='val')
    ax2.set_xlabel(r'Epoch')
    ax2.set_ylabel(r'Accuracy')
    ax2.legend()


    # Plot the recall
    ax3.plot(np.sqrt(history.history['recall_m']), 'r', label='train')
    ax3.plot(np.sqrt(history.history['val_recall_m']), 'b' ,label='val')
    ax3.set_xlabel(r'Epoch')
    ax3.set_ylabel(r'Recall')
    ax3.legend()


    fig.savefig('stats.png')


train_X, train_Y, valid_X, valid_Y, test_X, test_Y = getData('stockData.csv', True)

model, history = trainModel(train_X, train_Y, valid_X, valid_Y)

pickle.dump(model, open('NN_model.pkl','wb'))

testModel(model, test_X, test_Y)

#plotMetrics(history)