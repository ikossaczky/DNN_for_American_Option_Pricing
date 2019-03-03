import numpy as np

import tensorflow as tf
import keras.backend as K
import keras

def AmericanOptionLSTM(timesteps, LSTM=False, units=20):
    stock_prices = keras.layers.Input(shape=(timesteps + 1, 1), name='Stock_prices')
    timepoints = keras.layers.Input(shape=(timesteps + 1, 1), name='Timepoints')

    x = stock_prices
    if LSTM:
        x = keras.layers.LSTM(units=units, return_sequences=True, name='LSTM1', activation='hard_sigmoid')(x)
    else:
        x = keras.layers.Conv1D(filters=units, kernel_size=1, name='FC1-Conv', activation='hard_sigmoid')(x)
    x = keras.layers.Concatenate(axis=2, name='ConcatTime')([x, timepoints])
    if LSTM:
        x = keras.layers.LSTM(units=1, return_sequences=True, name='LSTM2', activation='hard_sigmoid')(x)
    else:
        x = keras.layers.Conv1D(filters=int(units // 2), kernel_size=1, name='FC2-Conv0', activation='elu')(x)
        x = keras.layers.Conv1D(filters=int(units // 4), kernel_size=1, name='FC2-Conv1', activation='elu')(x)
        x = keras.layers.Conv1D(filters=1, kernel_size=1, name='FC2-Conv2', activation='hard_sigmoid')(x)
    # x=keras.layers.Conv1D(filters=2,kernel_size=1)(x)
    outputs = keras.layers.Concatenate(name='Output', axis=2)([x, stock_prices, timepoints])
    return keras.models.Model(inputs=[stock_prices, timepoints], outputs=outputs)