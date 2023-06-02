#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Aldo Pastore, Zheng Yuan
# Date created: 15/11/2021
# Date last modified: 02/06/2023
# Python Version: 3.9.13
# License: CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, SimpleRNNCell, RNN
# import tensorflow_addons as tfa


from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, SimpleRNNCell, RNN

class SiameseModel(Model):
    """
    Siamese RNN model for sequence comparison.
    

    Args:
        nHidden (int): Number of hidden units in the feedforward layer.
        nHiddenRNN (int): Number of hidden units in the RNN cells.
        nLayersRNN (int): Number of layers in the RNN.
        distance (str): Distance metric for computing similarity between sequences.
        bidirectional (bool): Flag indicating whether to use bidirectional RNN.
        LSTM (bool): Flag indicating whether to use LSTM cells instead of SimpleRNN cells.
        l1rnn (float): L1 regularization strength for RNN weights.
        l1ff (float): L1 regularization strength for feedforward layer weights.
        l2rnn (float): L2 regularization strength for RNN weights.
        l2ff (float): L2 regularization strength for feedforward layer weights.

    Attributes:
        nHidden (int): Number of hidden units in the feedforward layer.
        nLayersRNN (int): Number of layers in the RNN.
        l1rnn (float): L1 regularization strength for RNN weights.
        l2rnn (float): L2 regularization strength for RNN weights.
        l1ff (float): L1 regularization strength for feedforward layer weights.
        l2ff (float): L2 regularization strength for feedforward layer weights.
        randomInit (tf.keras.initializers.Initializer): Random uniform initializer for weights.
        masking_layer (tf.keras.layers.Masking): Masking layer to handle variable-length sequences.
        cells (list): List of RNN cells (SimpleRNN or LSTM).
        RNN (tf.keras.layers.RNN or tf.keras.layers.Bidirectional): RNN layer (unidirectional or bidirectional).
        Dropout (tf.keras.layers.Dropout): Dropout layer for regularization.
        batchNorm (tf.keras.layers.BatchNormalization): Batch normalization layer.
        FeedForward (tf.keras.layers.Dense): Feedforward layer.
        distanceLayer (tf.keras.layers.Lambda): Lambda layer for computing distance/similarity between sequences.

    """


    def __init__(self, 
                 nHidden, 
                 nHiddenRNN,
                 nLayersRNN,
                 distance="cosSim", 
                 bidirectional=True, 
                 LSTM=False, 
                 l1rnn=0,
                 l1ff=0, 
                 l2rnn=0, 
                 l2ff=0):
        super(SiameseModel, self).__init__()

        self.nHidden = nHidden
        self.nLayersRNN = nLayersRNN
        self.l1rnn = l1rnn
        self.l2rnn = l2rnn
        self.l1ff = l1ff
        self.l2ff = l2ff
        self.randomInit = tf.random_uniform_initializer(minval=-0.3, maxval=0.3)

        self.masking_layer = tf.keras.layers.Masking(mask_value=0.0)

        if LSTM:
            # LSTM cells
            self.cells = [tf.keras.layers.LSTMCell(nHiddenRNN, 
                                                   activation='tanh', 
                                                   kernel_initializer=self.randomInit,
                                                   recurrent_initializer=self.randomInit,
                                                   recurrent_regularizer=tf.keras.regularizers.l1_l2(self.l1rnn, self.l2rnn),
                                                   kernel_regularizer=tf.keras.regularizers.l1_l2(self.l1rnn, self.l2rnn)) 
                            for i in range(0, nLayersRNN)]
        else:
            # SimpleRNN cells
            self.cells = [SimpleRNNCell(nHiddenRNN, 
                                        activation='tanh', 
                                        kernel_initializer=self.randomInit,
                                        recurrent_initializer=self.randomInit,
                                        recurrent_regularizer=tf.keras.regularizers.l1_l2(self.l1rnn, self.l2rnn),
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(self.l1rnn, self.l2rnn)) 
                            for i in range(0, nLayersRNN)]

        if bidirectional:
            # Bidirectional RNN
            self.RNN = tf.keras.layers.Bidirectional(RNN(self.cells, return_sequences=False, return_state=False))
        else:
            # Unidirectional RNN
            self.RNN = RNN(self.cells, return_sequences=False, return_state=False)

        # Dropout to regularize
        self.Dropout = tf.keras.layers.Dropout(0.0)
        # Batch normalization
        self.batchNorm = tf.keras.layers.BatchNormalization()
        # Feedforward layer
        self.FeedForward = Dense(self.nHidden, 
                                 activation='sigmoid', 
                                 kernel_initializer=self.randomInit,
                                 kernel_regularizer=tf.keras.regularizers.l1_l2(self.l1ff, self.l2ff))

        if distance == "cosSim":
            # Cosine similarity
            self.distanceLayer = tf.keras.layers.Lambda(
                lambda tensor: tf.multiply(-1.0, tf.keras.losses.cosine_similarity(tensor[0], tensor[-1], axis=-1)))
        elif distance == "l2":
            # L2 distance
            self.distanceLayer = tf.keras.layers.Lambda(
                lambda tensor: tf.add(1.0, tf.multiply(-1.0, tf.math.squared_difference(tensor[0], tensor[-1]))))  # tf.add(1.0, tf.multiply(-1.0, IS THE LOGICAL NOT (WE WANT OUTPUT 1 WHEN DISTANCE IS CLOSE TO 0
        elif distance == "l2_simple":
            # Simple L2 distance
            self.distanceLayer = tf.keras.layers.Lambda(
                lambda tensor: tf.reduce_mean(tf.math.squared_difference(tensor[0], tensor[-1]), axis=-1))

        elif distance == "l1":
            # L1 distance
            self.distanceLayer = tf.keras.layers.Lambda(
                lambda tensor: tf.add(1.0, tf.multiply(-1.0, tf.abs(tf.subtract(tensor[0], tensor[-1])))))

        elif distance == "Manhattan":
            # Manhattan distance
            self.distanceLayer = tf.keras.layers.Lambda(
                lambda tensor: tf.exp(tf.multiply(-10.0, tf.abs(tf.subtract(tensor[0], tensor[-1])))))

    def call(self, x, training=True):
        """
        Perform a forward pass through the Siamese model.

        Args:
            x (tuple): Input sequences (x1, x2) for comparison.
            training (bool): Flag indicating whether the model is in training mode.

        Returns:
            out (tf.Tensor): Output representing the distance/similarity between the input sequences.

        """

        x1, x2 = x
        x1 = self.masking_layer(x1)
        x2 = self.masking_layer(x2)

        x1 = self.RNN(x1)
        x2 = self.RNN(x2)

        x1 = self.batchNorm(x1)
        x2 = self.batchNorm(x2)

        x1 = self.Dropout(x1)
        x2 = self.Dropout(x2)

        x1 = self.FeedForward(x1)
        x2 = self.FeedForward(x2)

        # ebd1 = x1
        # ebd2 = x2

        out = self.distanceLayer((x1, x2))

        # return out, ebd1, ebd2
        return out
    

def binaryAccuracy(y_true, y_pred):

    '''
    This function calculates the accuracy of the model.
    
    :param: y_true: the true labels
    :param: y_pred: the predicted labels
    
    '''
    binary_predictions = tf.round(y_pred)
    return tf.cast(tf.equal(binary_predictions, y_true), dtype=tf.int32)

