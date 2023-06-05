#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Aldo Pastore, Zheng Yuan
# Date created: 15/11/2021
# Date last modified: 02/06/2023
# Python Version: 3.9.13
# License: CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)

import os
from Utilities import *
import shutil
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
import datetime
import argparse
from Config import *

#####Script Flags#############################################################################################
parser = argparse.ArgumentParser(description="Set configaration to train/retrain a Siamese network")

data_mode = parser.add_mutually_exclusive_group()
data_mode.add_argument("-f", '--french', action="store_true", help="Choose the French dataset")
data_mode.add_argument("-i", '--italian', action="store_true",  help="Choose the Italian dataset")
data_mode.add_argument('-a', '--all', action="store_true", help="Choose all datasets")

args = parser.parse_args()
##############################################################################################################


start = time.time()
ct = datetime.datetime.now()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

MODE = input('Please name your training (e.g. TR_220723_FR_D-FR-Train):  ')


##### NET PARAMETERS
Nepochs = 100
batchSize = 1
e_stop = 3
Nhiddens = [50]
NhiddensRNN = [50]
NlayersRNN = [1]
bidirectional = True
lstm = False


def setParams(Nep, bSize, eStop, nH=[50], nHrnn=[50], nLrnn=[1], bidir=True, LSTM=False):

    '''
    Set the parameters of the network.
    
    params: Nepochs: int--number of epochs.
    params: bSize: int--batch size.
    params: eStop: int--number of epochs with no improvement after which training will be stopped.
    params: nH: list--number of hidden units in the hidden layers.
    params: nHrnn: list--number of hidden units in the RNN layers.
    params: nLrnn: list--number of RNN layers.
    params: bidir: bool--whether to use bidirectional RNNs.
    params: LSTM: bool--whether to use LSTM RNNs.
    '''

    global Nepochs, batchSize, e_stop, Nhiddens, NhiddensRNN, NlayersRNN, bidirectional, lstm

    Nepochs = Nep
    batchSize = bSize
    e_stop = eStop

    ##### NET PARAMETERS
    Nhiddens = nH
    NhiddensRNN = nHrnn
    NlayersRNN = nLrnn
    bidirectional = bidir
    lstm = LSTM


def netConfig(bidirectional, lstm, networkType, distance, mode=MODE):

    '''
    Set the experiment name based network configuration.

    params: bidirectional: bool--whether to use bidirectional RNNs.
    params: lstm: bool--whether to use LSTM RNNs.
    params: networkType: int--type id of network.
    params: distance: str--type of distance metric.
    params: mode: str--name of the training.

    '''

    if bidirectional and lstm:
        net_kind = "bidirectional_LSTM_"
    elif bidirectional and not lstm:
        net_kind = "bidirectional_"
    else:
        net_kind = ""

    if networkType == 0:
        netName = "SiameseNet_" + distance + "_" + net_kind
        ExpNames = ["Hrnn" + str(NhiddensRNN[i]) + "H" + str(Nhiddens[i]) + "L" + str(NlayersRNN[i]) + netName for i in
                    range(0, len(NhiddensRNN))]

    elif networkType == 1:
        netName = "SiameseNet_onlyRNN_" + distance + "_" + net_kind
        ExpNames = ["Hrnn" + str(NhiddensRNN[i]) + "L" + str(NlayersRNN[i]) + netName for i in
                    range(0, len(NhiddensRNN))]

    elif networkType == 2:
        netName = "SiameseNetL2_" + net_kind
        ExpNames = ["Hrnn" + str(NhiddensRNN[i]) + "H" + str(Nhiddens[i]) + "L" + str(NlayersRNN[i]) + netName for i in
                    range(0, len(NhiddensRNN))]

    elif networkType == 3:
        netName = "SiameseNetL1_" + net_kind
        ExpNames = ["Hrnn" + str(NhiddensRNN[i]) + "H" + str(Nhiddens[i]) + "L" + str(NlayersRNN[i]) + netName for i in
                    range(0, len(NhiddensRNN))]

    ExpNames = [mode + "_" + name for name in ExpNames]

    return ExpNames




##### TRAINING #####

if __name__ == "__main__":

    networkType = 0  # 0 is rnn+ff+distance; 1 is rnn+distance; 2 is rnn+l2+ff; 3 is rnn+l1+ff
    distance = "cosSim"  # possible choices are "cosSim", "Manhattan", "l1", "l2"
    setParams(Nep=100, bSize=1, eStop=3, nH=[50], nHrnn=[50], nLrnn=[1], bidir=True, LSTM=False)

    ExpNames = netConfig(bidirectional, lstm, networkType, distance, MODE)

    l1rnns, l1ffs, l2rnns, l2ffs = getRegs(l1rnn=0.0009027863372380549, l1ff=0.0002867489040629811, l2rnn=0.0, l2ff=0.0,
                                        fflow=-100, ffHigh=-100, rnnLow=0, rnnHigh=0, N=1, l1=False, l2=False)
    l1rnns[0] = 0.0009027863372380549
    l1ffs[0] = 0.0002867489040629811

    nH = Nhiddens[0]
    nL = NlayersRNN[0]
    nHrnn = NhiddensRNN[0]
    expName = ExpNames[0]
    l1rnn, l1ff, l2rnn, l2ff = l1rnns[0], l1ffs[0], l2rnns[0], l2ffs[0]

    losses_min = []
    accs_min = []
    lr = 0.005


    siamesenet, expName = getNet(expName, networkType, nHrnn, nL, nH, distance, bidirectional, lstm, l1rnn, l1ff, l2rnn,
                                l2ff)

    if args.french:
        couples_p_FR, couples_n_FR, sents_train_p_FR, sents_train_n_FR, sents_val_p_FR, sents_val_n_FR = get_couple_sent_combs(SUBS_FR, COUPLES_FR, 2, 4)    
        # divide FR data into train and test
        filename_train_p_FR = get_data_filename(couples_p_FR, sents_train_p_FR, 'P', 'solo')
        filename_train_p_FR = [DATA_PATH_FR + filename for filename in filename_train_p_FR]
        filename_train_n_FR = get_data_filename(couples_n_FR, sents_train_n_FR, 'N', 'solo')
        filename_train_n_FR = [DATA_PATH_FR + filename for filename in filename_train_n_FR]
        
        filename_val_p_FR = get_data_filename(couples_p_FR, sents_val_p_FR, 'P', 'solo')
        filename_val_p_FR = [DATA_PATH_FR + filename for filename in filename_val_p_FR]
        filename_val_n_FR = get_data_filename(couples_n_FR, sents_val_n_FR, 'N', 'solo')
        filename_val_n_FR = [DATA_PATH_FR + filename for filename in filename_val_n_FR]

        filename_train_pos = filename_train_p_FR
        filename_train_neg = filename_train_n_FR
        filename_val_pos = filename_val_p_FR
        filename_val_neg = filename_val_n_FR

        print('French data loaded')

    elif args.italian:
        couples_p_ITA, couples_n_ITA, sents_train_p_ITA, sents_train_n_ITA, sents_val_p_ITA, sents_val_n_ITA = get_couple_sent_combs(SUBS_ITA, COUPLES_ITA, 2, 4)

        # divide ITA data into train and test
        filename_train_p_ITA = get_data_filename(couples_p_ITA, sents_train_p_ITA, 'P', 'solo')
        filename_train_p_ITA = [DATA_PATH_ITA + filename for filename in filename_train_p_ITA]
        filename_train_n_ITA = get_data_filename(couples_n_ITA, sents_train_n_ITA, 'N', 'solo')
        filename_train_n_ITA = [DATA_PATH_ITA + filename for filename in filename_train_n_ITA]

        filename_val_p_ITA = get_data_filename(couples_p_ITA, sents_val_p_ITA, 'P', 'solo')
        filename_val_p_ITA = [DATA_PATH_ITA + filename for filename in filename_val_p_ITA]
        filename_val_n_ITA = get_data_filename(couples_n_ITA, sents_val_n_ITA, 'N', 'solo')
        filename_val_n_ITA = [DATA_PATH_ITA + filename for filename in filename_val_n_ITA]

        filename_train_pos = filename_train_p_ITA
        filename_train_neg = filename_train_n_ITA
        filename_val_pos = filename_val_p_ITA
        filename_val_neg = filename_val_n_ITA

        print('Italian data loaded')

    elif args.all:
        couples_p_FR, couples_n_FR, sents_train_p_FR, sents_train_n_FR, sents_val_p_FR, sents_val_n_FR = get_couple_sent_combs(SUBS_FR, COUPLES_FR, 2, 4)    
        # divide FR data into train and test
        filename_train_p_FR = get_data_filename(couples_p_FR, sents_train_p_FR, 'P', 'solo')
        filename_train_p_FR = [DATA_PATH_FR + filename for filename in filename_train_p_FR]
        filename_train_n_FR = get_data_filename(couples_n_FR, sents_train_n_FR, 'N', 'solo')
        filename_train_n_FR = [DATA_PATH_FR + filename for filename in filename_train_n_FR]

        filename_val_p_FR = get_data_filename(couples_p_FR, sents_val_p_FR, 'P', 'solo')
        filename_val_p_FR = [DATA_PATH_FR + filename for filename in filename_val_p_FR]
        filename_val_n_FR = get_data_filename(couples_n_FR, sents_val_n_FR, 'N', 'solo')
        filename_val_n_FR = [DATA_PATH_FR + filename for filename in filename_val_n_FR]
        
        couples_p_ITA, couples_n_ITA, sents_train_p_ITA, sents_train_n_ITA, sents_val_p_ITA, sents_val_n_ITA = get_couple_sent_combs(SUBS_ITA, COUPLES_ITA, 2, 4)
        filename_train_p_ITA = get_data_filename(couples_p_ITA, sents_train_p_ITA, 'P', 'solo')
        filename_train_p_ITA = [DATA_PATH_ITA + filename for filename in filename_train_p_ITA]
        filename_train_n_ITA = get_data_filename(couples_n_ITA, sents_train_n_ITA, 'N', 'solo')
        filename_train_n_ITA = [DATA_PATH_ITA + filename for filename in filename_train_n_ITA]

        filename_val_p_ITA = get_data_filename(couples_p_ITA, sents_val_p_ITA, 'P', 'solo')
        filename_val_p_ITA = [DATA_PATH_ITA + filename for filename in filename_val_p_ITA]
        filename_val_n_ITA = get_data_filename(couples_n_ITA, sents_val_n_ITA, 'N', 'solo')
        filename_val_n_ITA = [DATA_PATH_ITA + filename for filename in filename_val_n_ITA]

        filename_train_pos = filename_train_p_FR + filename_train_p_ITA
        filename_train_neg = filename_train_n_FR + filename_train_n_ITA
        filename_val_pos = filename_val_p_FR + filename_val_p_ITA
        filename_val_neg = filename_val_n_FR + filename_val_n_ITA

        print('All data loaded')
    

    else:
        print('please choose french or italian with flag --italian, -i or --french, -f')
        exit()
    
    checkpoints_dir, tensorboard_dir, traininglog_dir = check_saver(PATH_SAVE_MODEL, expName)

    best_run_model = ""
    best_run_loss = 10000
    best_run_accuracy = 0

    train_losses = []
    val_losses = []

    train_acc = []
    train_acc_pos = []
    train_acc_neg = []

    val_acc = []
    val_acc_pos = []
    val_acc_neg = []

    loss_min = 100000
    acc_of_loss_min = 0


    optimizer = tf.keras.optimizers.Adam(lr=lr)
    siamesenet.compile(optimizer=optimizer, loss=tf.losses.binary_crossentropy, metrics=[binaryAccuracy])

    # Train
    for epoch in range(1, Nepochs + 1):
        count = 0
        random.seed(epoch)
        filename_train_neg_sub = random.sample(filename_train_neg, len(filename_train_pos))
        filename_train = filename_train_pos + filename_train_neg_sub
        filename_val = filename_val_pos + filename_val_neg
        print(f'Traning set size: {len(filename_train)}')
        print(f'Validation set size: {len(filename_val)}')

        random.shuffle(filename_train)
        train_dataset = iter(get_iterable(filename_train, batchSize))

        # learning rate decay
        if epoch > 1:
            lr = lr * 0.8 
            optimizer.learning_rate.assign(lr)
            print(optimizer.learning_rate)


        for _, _, x1, x2, target in train_dataset:

            if count % 2 == 0 and count != 0:

                loss, acc_mean, acc_pos, acc_neg = model_eval(siamesenet, filename_val_pos,
                                                            filename_val_neg, batch=batchSize)
                print('Epoch: ' + str(epoch) + ' - Step: ' + str(count + 1))
                print('Loss (binary crossentropy)  is ' + str(loss))
                print('Validation Accuracy is ' + str(acc_mean))
                print('Validation Accuracy positive is ' + str(acc_pos))
                print('Validation Accuracy negative is ' + str(acc_neg))

                val_acc.append(acc_mean)
                val_acc_pos.append(acc_pos)
                val_acc_neg.append(acc_neg)
                val_losses.append(loss)

                np.save(traininglog_dir + 'val_acc.npy', np.asarray(val_acc))
                np.save(traininglog_dir + 'val_acc_pos.npy', np.asarray(val_acc_pos))
                np.save(traininglog_dir + 'val_acc_neg.npy', np.asarray(val_acc_neg))
                np.save(traininglog_dir + 'val_loss.npy', np.asarray(val_losses))

                if loss < loss_min:  # Save the model if validation loss decreased
                    if os.path.isdir(checkpoints_dir):
                        shutil.rmtree(checkpoints_dir)
                        os.mkdir(checkpoints_dir)
                    loss_min = loss
                    acc_of_loss_min = acc_mean
                    siamesenet.save(checkpoints_dir, save_format='tf')

                print('Minimum LOSS is ' + str(loss_min) + ' with accuracy ' + str(acc_of_loss_min))

            log = siamesenet.fit(x=(x1, x2), y=target, batch_size=batchSize, epochs=1)
            count = count + 1

            train_acc.append(log.history['binaryAccuracy'])
            train_losses.append(log.history['loss'])
            np.save(traininglog_dir + 'train_acc.npy', np.asarray(train_acc))
            np.save(traininglog_dir + 'train_loss.npy', np.asarray(train_losses))


    plt.figure(figsize=(6, 10))
    plt.subplot(2, 1, 1)
    plt.plot(range(len(train_losses)), train_losses, 'b', label='Train loss')
    plt.plot([i*10 for i in range(1, len(val_losses)+1)], val_losses, 'g', label='Valid loss')
    plt.title('Train and Validation loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(range(len(train_acc)), train_acc, 'b', label='Train acc')
    plt.plot([i*10 for i in range(1, len(val_acc)+1)], val_acc, 'g', label='Valid acc')
    
    plt.title('Train and valid acc')
    plt.xlabel('Steps')
    plt.ylabel('Acc')
    plt.legend()

    plt.savefig(traininglog_dir + "train-val_result.pdf")
    plt.close()


    end = time.time()
    print('TRAINING FINISHED, RUN TIME: ', end-start)
