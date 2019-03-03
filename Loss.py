import tensorflow as tf
import keras.backend as K
import keras

def profit_loss(strike=5, interest_rate=0.05, T=1, option_type='call'):

    def profit_loss_(dummy_target, statedata):

        # probability that we will exercise at the given time if we still haven't:
        exerprobs= statedata[:, :, 0]

        # stockprices:
        stockprices = statedata[:, :, 1]

        # timepoints:
        timepoints = statedata[:, :, 2]/T

        # probabilty that we havent exercised the option yet:
        pshift = K.concatenate([K.zeros_like(exerprobs[:, 0:1]), exerprobs[:, :-1]])
        one_minus_pshift = 1 - pshift
        prob_notexercisedbefore = K.cumprod(one_minus_pshift, axis=1)

        # probability of exercising the option at the given time:
        willexerprob = exerprobs * prob_notexercisedbefore

        # profit for exercising at current time:
        if option_type=='call':
            profit = K.maximum(stockprices - strike, 0)
        else:
            profit = K.maximum(strike - stockprices, 0)

        # discounting profit:
        profit = K.exp(-interest_rate * timepoints) * profit

        # average profit:
        ap = K.sum(profit * willexerprob, axis=1)
        return -K.mean(ap)

    return profit_loss_