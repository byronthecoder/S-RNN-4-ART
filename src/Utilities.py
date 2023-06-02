#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Aldo Pastore, Zheng Yuan
# Date created: 15/11/2021
# Date last modified: 02/06/2023
# Python Version: 3.9.13
# License: CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)

import numpy as np
import glob
import os
import random
import gc
from Siamese_models import *
from matplotlib import pyplot as pp
import re
from itertools import combinations, product


def get_couple_sent_combs(subs, couples, nSentTrain, nSentAll):
    
    '''get couple and sentence combinations for building training set and validation set.
    
    params: subs: list of speakers.
    params: couples: list of couples.
    params: nSentTrain: upper bound number of sentences to use in training set.
    params: nSentAll: total number of sentences in the dataset.

    '''
    
    couples_p =[[str(sub), str(sub)] for sub in subs]
    couples_n = couples

    sents_train_p = np.asarray(list(combinations(range(1, nSentTrain+1), 2))) 

    sents_train_n = np.asarray(list(product(range(1, nSentTrain+1), range(1, nSentTrain+1))))

    sents_val_p = np.asarray(list(combinations(range(nSentTrain+1, nSentAll+1), 2))) 

    sents_val_n = np.asarray(list(product(range(nSentTrain+1, nSentAll+1), range(nSentTrain+1, nSentAll+1))))
    
    return couples_p, couples_n, sents_train_p, sents_train_n, sents_val_p, sents_val_n
    

def get_data_filename(couples, sent_combs, name, sess):
    
    '''Get tfrecords dataset filenames.
    
    params: couples: list of couples.
    params: sent_combs: list of sentence combinations.
    params: name: lable of the dataset.
    params: sess: session name.    
    '''
    
    filenames = []
    
    for couple in couples:
        for sents in sent_combs:
            sub1 = couple[0]
            sub2 = couple[1]
            sent1 = str(sents[0])
            sent2 = str(sents[1])

            outfilename = str(name + 
                              '-s' + sub1 + '_sent' + sent1 + '_' + sess + 
                              '-s' + sub2 + '_sent' + sent2 + '_' + sess + 
                              '.tfrecords')
            filenames.append(outfilename)
            
    return filenames


def check_saver(pSave, ExpNum):

    '''
    Check if saver folder exists, if not, create it.
    
    params: pSave: path to save folder.
    params: ExpNum: experiment number/name.

    '''
    checkpoints_dir = pSave + str(ExpNum) + '/checkpoints/'
    tensorboard_dir = pSave + str(ExpNum) + '/tensorboard/'
    traininglog_dir = pSave + str(ExpNum) + '/training_logs/'

    if not os.path.exists(pSave):
        os.makedirs(pSave)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(traininglog_dir):
        os.makedirs(traininglog_dir)

    return checkpoints_dir, tensorboard_dir, traininglog_dir



def read_my_file_format(serialized_example, feat_dimension=39):

    '''
    Read tfrecords dataset.
    
    '''
    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(serialized_example,
                                                                          context_features={
                                                                              "feat1_length": tf.io.FixedLenFeature([],
                                                                                                                    dtype=tf.int64),
                                                                              "feat2_length": tf.io.FixedLenFeature([],
                                                                                                                    dtype=tf.int64)},
                                                                          sequence_features={
                                                                              "audio1_feat": tf.io.FixedLenSequenceFeature(
                                                                                  [feat_dimension], dtype=tf.float32),
                                                                              "audio2_feat": tf.io.FixedLenSequenceFeature(
                                                                                  [feat_dimension], dtype=tf.float32),
                                                                              "labels": tf.io.FixedLenSequenceFeature(
                                                                                  [], dtype=tf.int64)}
                                                                          )
    return tf.dtypes.cast(context_parsed['feat1_length'], tf.int32), tf.dtypes.cast(context_parsed['feat2_length'],
                                                                                    tf.int32), sequence_parsed[
               'audio1_feat'], sequence_parsed['audio2_feat'], tf.dtypes.cast(sequence_parsed['labels'], tf.int32)


def get_data(filenames, shuffle):

    '''
    Map tfrecords dataset to tf.data.Dataset.
    
    '''

    if shuffle: random.shuffle(filenames)
    data = tf.data.TFRecordDataset(filenames)  # num_parallel_reads=CPU_COUNT)
    data = data.map(read_my_file_format)  # ,num_parallel_calls=CPU_COUNT)
    return data


def get_iterable(filenames, batch_size=1, shuffle=True):

    '''
    pad and batch the dataset.
    
    '''
    data = get_data(filenames, shuffle)
    batch_data = data.padded_batch(batch_size, padded_shapes=([], [], [None, None], [None, None], [None]),
                                   drop_remainder=True)

    return batch_data


def get_oneSet(filenames, shuffle=False):

    '''
    Get one set of data, not batched.

    '''
    set = iter(get_iterable(filenames, len(filenames), shuffle))
    L1, L2, seq1, seq2, labels = next(set)
    return seq1, seq2, L1, L2, labels

def getDataset(filenames_train_pos, filenames_train_neg, filenames_val_pos, filenames_val_neg,
               len_trPos=1, len_trNeg=1, len_valPos=1, len_valNeg=1, shuffle=True, get_all1=False, get_all2=False):

    '''
    Another implementation of tfrecord dataset. Get all the data in one set.

    '''
    if get_all1 == True:
        len_trPos = len(filenames_train_pos)
        len_trNeg = len(filenames_train_neg)
    else:
        len_trPos = len_trPos * len(filenames_train_pos)
        len_trNeg = len_trNeg * len(filenames_train_pos)

    if get_all2 == True:
        len_valPos = len(filenames_val_pos)
        len_valNeg = len(filenames_val_neg)
    else:
        len_valPos = len_valPos * len(filenames_val_pos)
        len_valNeg = len_valNeg * len(filenames_val_pos)

    pos_datset = iter(get_iterable(filenames_train_pos, len_trPos, shuffle))
    neg_dataset = iter(get_iterable(filenames_train_neg, len_trNeg, shuffle))

    pos_val_dataset = iter(get_iterable(filenames_val_pos, len_valPos, shuffle))
    neg_val_dataset = iter(get_iterable(filenames_val_neg, len_valNeg, shuffle))

    LseqP1_val, LseqP2_val, pos1_val, pos2_val, labels_pos_val = next(pos_val_dataset)
    LseqN1_val, LseqN2_val, neg1_val, neg2_val, labels_neg_val = next(neg_val_dataset)

    LseqP1, LseqP2, pos1, pos2, labels_pos = next(pos_datset)
    LseqN1, LseqN2, neg1, neg2, labels_neg = next(neg_dataset)

    data1 = np.concatenate((pos1, neg1))
    data2 = np.concatenate((pos2, neg2))
    labels = tf.concat((labels_pos, labels_neg), axis=0)

    data1_val = np.concatenate((pos1_val, neg1_val))
    data2_val = np.concatenate((pos2_val, neg2_val))
    labels_val = tf.concat((labels_pos_val, labels_neg_val), axis=0)

    return data1, data2, labels, data1_val, data2_val, labels_val, labels_pos, labels_neg, pos1, pos2, neg1, neg2, labels_pos_val, labels_neg_val, pos1_val, pos2_val, neg1_val, neg2_val


def model_eval(model, filename_val_pos, filename_val_neg, filename_test_pos=None, filename_test_neg=None, batch=1):

    '''
    Evaluate the model on the test set.

    :param model: the model to evaluate.
    :param filename_val_pos: list of the filename of the positive validation set.
    :param filename_val_neg: list of the filename of the negative validation set.
    :param filename_test_pos: list of the filename of the positive test set.
    :param filename_test_neg: list of the filename of the negative test set.
    :param batch: the batch size.
    
    '''

    batch_p = batch
    batch_n = batch

    if len(filename_val_pos) < batch:
        batch_p = len(filename_val_pos)

    if len(filename_val_neg) < batch:
        batch_n = len(filename_val_neg)


    pos_val_ds = iter(get_iterable(filename_val_pos, batch_p))
    neg_val_ds = iter(get_iterable(filename_val_neg, batch_n))

    pos_res = []
    neg_res = []

    for _, _, x1_val, x2_val, target_val in pos_val_ds:
        pos_res.append(model.evaluate((x1_val, x2_val), target_val, batch_size=len(x1_val), verbose=0))
        x1_val, x2_val, target_val = None, None, None
        gc.collect()

    for _, _, x1_val, x2_val, target_val in neg_val_ds:
        neg_res.append(model.evaluate((x1_val, x2_val), target_val, batch_size=len(x1_val), verbose=0))
        x1_val, x2_val, target_val = None, None, None
        gc.collect()

    loss_pos = np.mean(np.asarray([pos_r[0] for pos_r in pos_res]))
    accuracy_pos = np.mean(np.asarray([pos_r[1] for pos_r in pos_res]))

    loss_neg = np.mean(np.asarray([neg_r[0] for neg_r in neg_res]))
    accuracy_neg = np.mean(np.asarray([neg_r[1] for neg_r in neg_res]))

    accuracy = (accuracy_pos + accuracy_neg) / 2
    loss = (loss_pos + loss_neg) / 2

    if filename_test_pos != None and filename_test_neg != None:
        pos_test_ds = iter(get_iterable(filename_test_pos, batch))
        neg_test_ds = iter(get_iterable(filename_test_neg, batch))

        pos_res = []
        neg_res = []

        for _, _, x1_test, x2_test, target_test in pos_test_ds:
            pos_res.append(model.evaluate((x1_test, x2_test), target_test, batch_size=len(x1_test), verbose=0))
            x1_test, x2_test, target_test = None, None, None
            gc.collect()

        for _, _, x1_test, x2_test, target_test in neg_test_ds:
            neg_res.append(model.evaluate((x1_test, x2_test), target_test, batch_size=len(x1_test), verbose=0))
            x1_test, x2_test, target_test = None, None, None
            gc.collect()

        loss_pos_test = np.mean(np.asarray([pos_r[0] for pos_r in pos_res]))
        accuracy_pos_test = np.mean(np.asarray([pos_r[1] for pos_r in pos_res]))

        loss_neg_test = np.mean(np.asarray([neg_r[0] for neg_r in neg_res]))
        accuracy_neg_test = np.mean(np.asarray([neg_r[1] for neg_r in neg_res]))

        accuracy_test = (accuracy_pos_test + accuracy_neg_test) / 2
        loss_test = (loss_pos_test + loss_neg_test) / 2

        return loss, accuracy, accuracy_pos, accuracy_neg, loss_test, accuracy_test, accuracy_pos_test, accuracy_neg_test

    else:
        return loss, accuracy, accuracy_pos, accuracy_neg

def model_eval_one_class(model, filename_val, filename_test=None, batch=1):

    '''
    Evaluate the model on the test set of one class.'''


    if len(filename_val) < batch:
        batch = len(filename_val)

    val_ds = iter(get_iterable(filename_val, batch))

    res = []

    for _, _, x1_val, x2_val, target_val in val_ds:
        res.append(model.evaluate((x1_val, x2_val), target_val, batch_size=len(x1_val), verbose=0))
        x1_val, x2_val, target_val = None, None, None
        gc.collect()

    loss = np.mean(np.asarray([r[0] for r in res]))
    accuracy = np.mean(np.asarray([r[1] for r in res]))

    # TEST
    if filename_test != None:
        test_ds = iter(get_iterable(filename_test, batch))

        res = []

        for _, _, x1_test, x2_test, target_test in test_ds:
            res.append(model.evaluate((x1_test, x2_test), target_test, batch_size=len(x1_test), verbose=0))
            x1_test, x2_test, target_test = None, None, None
            gc.collect()

        loss_test = np.mean(np.asarray([r[0] for r in res]))
        accuracy_test = np.mean(np.asarray([r[1] for r in res]))

        return loss, accuracy, loss_test, accuracy_test

    else:
        return loss, accuracy


def getRegs(l1rnn, l1ff, l2rnn, l2ff, fflow, ffHigh, rnnLow, rnnHigh, N, l1=True, l2=False):

    '''
    Returns a list of regularization parameters to be used in the model.

    params: l1rnn: L1 regularization parameters for RNNs.
    params: l1ff: L1 regularization parameters for FF layers.
    params: l2rnn: L2 regularization parameters for RNNs.
    params: l2ff: L2 regularization parameters for FF layers.
    params: fflow: Lower bound for FF regularization parameters.
    params: ffHigh: Upper bound for FF regularization parameters.
    params: rnnLow: Lower bound for RNN regularization parameters.
    params: rnnHigh: Upper bound for RNN regularization parameters.
    params: N: Number of regularization parameters to be returned.
    params: l1: Whether to employ L1 regularization.
    params: l2: Whether to employ L2 regularization.
    
    '''

    if l1: l1=0
    else : l1=1

    if l2: l2=0
    else: l2 =1

    rFF = np.random.uniform(low=fflow, high=ffHigh, size=(N,))
    rRNN = np.random.uniform(low=rnnLow, high=rnnHigh, size=(N,))

    rFF = np.power(10, rFF)
    rRNN = np.power(10, rRNN)

    l1rnns = rRNN + l1rnn - l1*rRNN
    l1ffs = rFF + l1ff - l1*rFF

    l2rnns = rRNN + l2rnn - l2*rRNN
    l2ffs = rFF + l2ff - l2*rFF

    return l1rnns, l1ffs, l2rnns, l2ffs



def getNet(expName, networkType, nHiddenRNN, nLayersRNN, nHidden=50,
            distance="cosSim",bidirectional=True, LSTM=False, l1rnn=0, l1ff=0, l2rnn=0, l2ff=0,add_regs=True):

    '''
    Returns a Siamese RNN model and the updated experiment name.
    
    params: expName: str--Name of the experiment.
    params: networkType: int--Type of network to be used.
    params: nHiddenRNN: int--Number of hidden units in the RNN.
    params: nLayersRNN: int--Number of layers in the RNN.
    params: nHidden: int--Number of hidden units in the FF.
    params: distance: str--Distance metric to be used.
    params: bidirectional: bool--Whether to use bidirectional RNNs.
    params: LSTM: bool--Whether to use LSTM RNNs.
    params: l1rnn: float--L1 regularization parameter for RNNs.
    params: l1ff: float--L1 regularization parameter for FF.
    params: l2rnn: float--L2 regularization parameter for RNNs.
    params: l2ff: float--L2 regularization parameter for FF.
    params: add_regs: bool--Whether to add regularization to the model.
    
    '''

    if networkType==0:
        net = SiameseModel(nHidden,nHiddenRNN,nLayersRNN, distance, bidirectional, LSTM, l1rnn, l1ff,l2rnn,l2ff)
        if add_regs: expName=expName+"_l1rnn" + str(l1rnn) + "_l1ff" + str(l1ff) + "_l2rnn"+str(l2rnn) + "_l2ff"+str(l2ff)


    elif networkType==1:
        net = SiameseModel_onlyRNN(nHiddenRNN, nLayersRNN, distance, bidirectional, LSTM, l1rnn, l2rnn)
        if add_regs: expName=expName+"_l1rnn" + str(l1rnn) + "_l2rnn"+str(l2rnn)


    elif networkType==2:
        net = SiameseModel_l2(nHiddenRNN, nLayersRNN, distance, bidirectional, LSTM, l1rnn, l2rnn)
        if add_regs: expName=expName+"_l1rnn" + str(l1rnn) + "_l1ff" + str(l1ff) + "_l2rnn"+str(l2rnn) + "_l2ff"+str(l2ff)


    elif networkType==3:
        net = SiameseModel_l1(nHidden,nHiddenRNN,nLayersRNN, distance, bidirectional, LSTM, l1rnn, l1ff,l2rnn,l2ff)
        if add_regs: expName=expName+"_l1rnn" + str(l1rnn) + "_l1ff" + str(l1ff) + "_l2rnn"+str(l2rnn) + "_l2ff"+str(l2ff)


    return net, expName

def plot_results(siamesenet,filenames_pos,filenames_neg, saver_dir, name=None, set="TEST"):

    '''
    Plots the results of the model.
    
    params: siamesenet: SiameseRNN-Model to be used.
    params: filenames_pos: list--List of positive filenames.
    params: filenames_neg: list--List of negative filenames.
    params: saver_dir: str--Directory to save the plots.
    params: name: str--Name keyword to be used in the plot title.
    params: set: str--DataSet to be used for the plot title. 

    '''


    ds_pos=iter(get_iterable(filenames_pos, len(filenames_pos)))
    ds_neg=iter(get_iterable(filenames_neg, len(filenames_neg)))

    for _, _, x1, x2, _ in ds_pos:
        results_pos=siamesenet((x1,x2))

    for _, _, x1, x2, _ in ds_neg:
        results_neg = siamesenet((x1, x2))

    pp.figure(figsize=(6, 10))
    pp.subplot(2, 1, 1)
    counts, bins = np.histogram(results_pos)
    pp.hist(bins[:-1], bins, weights=counts / len(results_pos), facecolor='r')
    pp.title(set + " - Similarity - Same speaker")
    pp.xlabel('similarity')
    pp.ylabel('percentage %')

    pp.subplot(2, 1, 2)
    counts, bins = np.histogram(results_neg)
    pp.hist(bins[:-1], bins, weights=counts / len(results_neg), facecolor='k')
    pp.title(set+ "- Similarity - Different speakers")
    pp.xlabel('similarity')
    pp.ylabel('percentage %')

    pp.savefig(saver_dir + set + name + "_result.pdf")
    pp.close()

    results_pos = np.asarray(results_pos)
    results_neg = np.asarray(results_neg)

    return results_pos, results_neg


