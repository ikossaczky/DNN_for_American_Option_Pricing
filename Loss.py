import sys, os, shutil, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend as K
import keras


def profit_loss(dummy_target, stockdata):
    # stockprices
    stockprice = stockdata[:, :, 1]  # inputdata
    # timestep
    timepoint = stockdata[:, :, 2]
    # probability that we will exercise at the given time if we still havent
    exerprob = stockdata[:, :, 0]  # K.squeeze(exerprob, axis=2)

    # probabilty that we havent exercised the option yet
    pshift = K.concatenate([K.zeros_like(exerprob[:, 0:1]), exerprob[:, :-1]])
    one_minus_pshift = 1 - pshift
    prob_notexercisedbefore = K.cumprod(one_minus_pshift, axis=1)

    # probability of using option at the given time:
    willexerprob = exerprob * prob_notexercisedbefore

    # profit for exercising at current time:
    profit = K.maximum(stockprice - 5, 0)

    # discounting profit:
    interest_rate = 0.05
    profit = K.exp(-interest_rate * timepoint) * profit

    # average profit:
    ap = K.sum(profit * willexerprob, axis=1)
    return -K.mean(ap)