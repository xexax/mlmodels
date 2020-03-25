import pandas as pd
import os
import numpy as np
import mdn
import math
import keras.regularizers as reg
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import Model
from keras.layers import Dense, Dropout, Input, LSTM, Concatenate
from keras.callbacks import History, EarlyStopping
from keras.models import model_from_json
from keras.regularizers import l2 
from keras.optimizers import Adam
from keras import backend as K   
from keras.models import model_from_json
from keras.constraints import max_norm
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import KFold
from matplotlib.pyplot import figure
from google.colab import files

class MDN(layers.Layer):
    """A Mixture Density Network Layer for Keras.
    This layer has a few tricks to avoid NaNs in the loss function when training:
        - Activation for variances is ELU + 1 + 1e-8 (to avoid very small values)
        - Mixture weights (pi) are trained in as logits, not in the softmax space.
    A loss function needs to be constructed with the same output dimension and number of mixtures.
    A sampling function is also provided to sample from distribution parametrised by the MDN outputs.
    """

    def __init__(self, output_dimension, num_mixtures, **kwargs):
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        with tf.name_scope('MDN'):
            self.mdn_mus = layers.Dense(self.num_mix * self.output_dim, name='mdn_mus')  # mix*output vals, no activation
            self.mdn_sigmas = layers.Dense(self.num_mix * self.output_dim, activation=elu_plus_one_plus_epsilon, name='mdn_sigmas')  # mix*output vals exp activation
            self.mdn_pi = layers.Dense(self.num_mix, name='mdn_pi')  # mix vals, logits
        super(MDN, self).__init__(**kwargs)

    def build(self, input_shape):
        with tf.name_scope('mus'):
            self.mdn_mus.build(input_shape)
        with tf.name_scope('sigmas'):
            self.mdn_sigmas.build(input_shape)
        with tf.name_scope('pis'):
            self.mdn_pi.build(input_shape)
        super(MDN, self).build(input_shape)

    @property
    def trainable_weights(self):
        return self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights

    def call(self, x, mask=None):
        with tf.name_scope('MDN'):
            mdn_out = layers.concatenate([self.mdn_mus(x),
                                          self.mdn_sigmas(x),
                                          self.mdn_pi(x)],
                                         name='mdn_outputs')
        return mdn_out

    def compute_output_shape(self, input_shape):
        """Returns output shape, showing the number of mixture parameters."""
        return (input_shape[0], (2 * self.output_dim * self.num_mix) + self.num_mix)

    def get_config(self):
        config = {
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix
        }
        base_config = super(MDN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent
    NaN in loss."""
    return K.elu(x) + 1 + K.epsilon()

def split_mixture_params(params, output_dim, num_mixes):
    """Splits up an array of mixture parameters into mus, sigmas, and pis
    depending on the number of mixtures and output dimension.
    Arguments:
    params -- the parameters of the mixture model
    output_dim -- the dimension of the normal models in the mixture model
    num_mixes -- the number of mixtures represented
    """
    mus = params[:num_mixes * output_dim]
    sigs = params[num_mixes * output_dim:2 * num_mixes * output_dim]
    pi_logits = params[-num_mixes:]
    return mus, sigs, pi_logits

def softmax(w, t=1.0):
    """Softmax function for a list or numpy array of logits. Also adjusts temperature.
    Arguments:
    w -- a list or numpy array of logits
    Keyword arguments:
    t -- the temperature for to adjust the distribution (default 1.0)
    """
    e = np.array(w) / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist

def sample_from_categorical(dist):
    """Samples from a categorical model PDF.
    Arguments:
    dist -- the parameters of the categorical model
    Returns:
    One sample from the categorical model, or -1 if sampling fails.
    """
    r = np.random.rand(1)  # uniform random number in [0,1]
    accumulate = 0
    for i in range(0, dist.size):
        accumulate += dist[i]
        if accumulate >= r:
            return i
    tf.logging.info('Error sampling categorical model.')
    return -1

def sample_from_output(params, output_dim, num_mixes, temp=1.0, sigma_temp=1.0):
    """Sample from an MDN output with temperature adjustment.
    This calculation is done outside of the Keras model using
    Numpy.
    Arguments:
    params -- the parameters of the mixture model
    output_dim -- the dimension of the normal models in the mixture model
    num_mixes -- the number of mixtures represented
    Keyword arguments:
    temp -- the temperature for sampling between mixture components (default 1.0)
    sigma_temp -- the temperature for sampling from the normal distribution (default 1.0)
    Returns:
    One sample from the the mixture model.
    """
    mus, sigs, pi_logits = split_mixture_params(params, output_dim, num_mixes)
    pis = softmax(pi_logits, t=temp)
    m = sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m * output_dim:(m + 1) * output_dim]
    sig_vector = sigs[m * output_dim:(m + 1) * output_dim]
    scale_matrix = np.identity(output_dim) * sig_vector  # scale matrix from diag
    cov_matrix = np.matmul(scale_matrix, scale_matrix.T)  # cov is scale squared.
    cov_matrix = cov_matrix * sigma_temp  # adjust for sigma temperature
    sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
    return sample


def get_mixture_loss_func(output_dim, num_mixes):
    """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
    # Construct a loss function with the right number of mixtures and outputs
    def mdn_loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + 
                                     num_mixes], name='reshape_ypreds')
        y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
        # Split the inputs into paramaters
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * output_dim,
                                                                         num_mixes],
                                             axis=-1, name='mdn_coef_split')
        # Construct the mixture models
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
        mixture = tfd.Mixture(cat=cat, components=coll)
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss

    # Actually return the loss function
    with tf.name_scope('MDN'):
        return mdn_loss_func


class Model:
  def __init__(self, model_pars=None, data_pars=None, compute_pars=None
               ):
    ### Model Structure        ################################
    # maxlen         = model_pars['maxlen']
    # max_features   = model_pars['max_features']
    # embedding_dims = model_pars['embedding_dims']

    # self.model = TextCNN(maxlen, max_features, embedding_dims).get_model()

    # self.model.compile(compute_pars['engine'],  # adam 
    # 	               compute_pars['loss'], 
    # 	               metrics= compute_pars['metrics'])

    lstm_h_list = model_pars["lstm_h_list"]
    OUTPUT_DIMS = model_pars["output_dims"]
    N_MIXES = model_pars["n_mixes"]
    learning_rate = model_pars["lr"]

    model = Sequential()
    for ind, hidden in enumerate(lstm_h_list):
    model.add(layers.LSTM(units=hidden, return_sequences=True, name=f"LSTM_{ind+1}", 
                    input_shape=(timesteps, 1), 
                    recurrent_regularizer=reg.l1_l2(l1=0.01, l2=0.01)))
    model.add(layers.Dense(dense_neuron, input_shape=(-1, lstm_h_list[-1]), 
                    activation='relu'))
    model.add(MDN(OUTPUT_DIMS, N_MIXES))
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                decay=0.0)
    model.compile(loss=get_mixture_loss_func(OUTPUT_DIMS, N_MIXES),
                optimizer=adam)
    self.model = model

def get_dataset(data_pars=None, **kw):
    path = pd.read_csv("../dataset/timeseries/milk.csv")
    df = pd.read_csv("../dataset/timeseries/milk.csv")

def fit(model, data_pars={}, compute_pars={}, out_pars={},   **kw):
    """
    """
    batch_size = compute_pars['batch_size']
    epochs = compute_pars['epochs']

    sess = None 
    Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)


    early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
    model.model.fit(Xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping],
            validation_data=(Xtest, ytest))

    return model, sess

if __name__ == "__main__":
    df = pd.read_csv("../dataset/timeseries/milk.csv")

