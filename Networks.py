import numpy as np

import tensorflow as tf
import keras.backend as K
import keras

def AmericanOptionNetwork_1(timesteps,
                            num_initial_features=20,
                            num_hidden_features=(10,5,2),
                            hidden_activation='elu',
                            concat_timepoints=(True, False, False)):

    # Input:
    stock_prices = keras.layers.Input(shape=(timesteps + 1, 1), name='Stock_prices')
    timepoints = keras.layers.Input(shape=(timesteps + 1, 1), name='Timepoints')
    x=stock_prices

    #  Projecting stock prices to features with values from [0,1]:
    x = keras.layers.Conv1D(filters=num_initial_features, kernel_size=1, name='FC-Conv_initial', activation='sigmoid')(x)

    # Processing features by a serie of FC layers:
    for k in range(len(num_hidden_features)):

        if concat_timepoints[k]:
            # Concatenate timepoints to the current layer:
            x = keras.layers.Concatenate(axis=2, name='ConcatTime_{}'.format(k))([x, timepoints])

        x = keras.layers.Conv1D(filters=num_hidden_features[k], kernel_size=1, name='FC-Conv_{}'.format(k),
                                activation=hidden_activation)(x)

    # Final layer outputing probabilities of exercising the option:
    exer_probs = keras.layers.Conv1D(filters=1, kernel_size=1, name='Exercise_Probabilty_Output', activation='sigmoid')(x)

    # Concatenating exercise probabilities with stockprices and timepoints for the output:
    concat_output = keras.layers.Concatenate(name='Concatenated_Output', axis=2)([exer_probs, stock_prices, timepoints])

    return keras.models.Model(inputs=[stock_prices, timepoints], outputs=concat_output)