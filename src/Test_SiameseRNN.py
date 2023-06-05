#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Aldo Pastore, Zheng Yuan
# Date created: 15/11/2021
# Date last modified: 02/06/2023
# Python Version: 3.9.13
# License: CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)

import os
from builtins import print
from Utilities import *
from Config import *
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import time
import sys
import argparse

#####Script Flags#############################################################################################

parser_test = argparse.ArgumentParser(description="Set configaration to test the model")

model_mode = parser_test.add_mutually_exclusive_group()
model_mode.add_argument('-f', '--french', action='store_true', help='Choose the French model')
model_mode.add_argument('-i', '--italian', action='store_true', help='Choose the Italian model')
model_mode.add_argument("-a", '--all', action="store_true",  help="Choose the French and Italian model")
model_mode.add_argument('-p', '--pretrained', action='store_true', help='Choose the pretrained VCTK + FR + ITA model')

parser_test.add_argument("-F", '--French', action="store_true", help="Choose the French dataset")
parser_test.add_argument("-I", '--Italian', action="store_true",  help="Choose the Italian dataset")
parser_test.add_argument("-A", '--All', action="store_true",  help="Choose the French and Italian datasets")

parser_test.add_argument("-v", "--validation", action="store_true", help="Do validation while testing")

test_mode = parser_test.add_mutually_exclusive_group()
test_mode.add_argument("-c", '--cross', action="store_true", help="Test the model with imitation data")
test_mode.add_argument("-s", '--same', action="store_true", help="Test the model with main0~main75 data")

args = parser_test.parse_args()
##############################################################################################################


start = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

MODE = input('Please name your test (e.g. TS_220723_FR_D-FR-Test):  ')


##### NET PARAMETERS
Nepochs = 1
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

def retrieveTrainedSiameseModel(pLoad, siamesenet):

    '''
    Load the trained model.

    :param pLoad: path to the model.
    :param siamesenet: configarated siamese network.
    
    '''

    if os.path.exists(pLoad + '/variables'):
        tf.keras.Model.load_weights(siamesenet, pLoad + '/variables/variables')
        print("Trained model loaded")
    else:
        sys.exit("Model NOT found!")
        

    return siamesenet


#####MAIN#############################################################################################################
if __name__ == '__main__':

    networkType = 0 # 0 is rnn+ff+distance; 1 is rnn+distance; 2 is rnn+l2+ff; 3 is rnn+l1+ff
    distance = "cosSim" # possible choices are "cosSim", "Manhattan", "l1", "l2"

    setParams(Nep=1, bSize=1, eStop=3, nH=[50], nHrnn=[50], nLrnn=[1], bidir=True, LSTM=False)

    ExpNames = netConfig(bidirectional, lstm, networkType, distance, mode=MODE)

    l1rnns, l1ffs, l2rnns, l2ffs = getRegs(l1rnn=0.0009027863372380549, l1ff=0.0002867489040629811, l2rnn=0.0, l2ff=0.0, fflow=-6, ffHigh=-6, rnnLow=-6, rnnHigh=-6, N=1, l1=True, l2=False)

    l1rnns[0]=0.0009027863372380549
    l1ffs[0]=0.0002867489040629811
    nH = Nhiddens[0]
    nL = NlayersRNN[0]
    nHrnn = NhiddensRNN[0]
    expName = ExpNames[0]
    l1rnn, l1ff, l2rnn, l2ff = l1rnns[0], l1ffs[0], l2rnns[0], l2ffs[0]


    siamesenet, expName = getNet(expName, networkType, nHrnn, nL, nH, distance, bidirectional, lstm, l1rnn, l1ff, l2rnn,l2ff)

    if not os.path.isdir(PATH_SAVE_RESULTS + expName): os.mkdir(PATH_SAVE_RESULTS + expName)
    resPath = PATH_SAVE_RESULTS + expName + '/'

    print("###Exp Settings###")
    print(expName)

    # lists to save results
    test_result_loss = []
    test_result_acc = []
    test_result_acc_pos = []
    test_result_acc_neg = []

    test_val_acc = []
    test_val_loss = []
    test_val_acc_pos = []
    test_val_acc_neg = []

    sim_pos_VAL = []
    sim_neg_VAL = []
    sim_pos_TEST = []
    sim_neg_TEST = []

    if args.french:
        
        model_path = PATH_MODEL_FR
        print('Trying to load French model')

    elif args.italian:
    
        model_path = PATH_MODEL_ITA
        print('Trying to load Italian model')

    elif args.all:

        model_path = PATH_MODEL_FR_ITA
        print('Trying to load Italian-French model')
    
    elif args.pretrained:
            model_path = PATH_MODEL_VCTK_FR_ITA
            print('Trying to load VCTK-Italian-French model') 
    else:
        print('No model specified, please use flag -french, -italian, -all to specify the model')
        sys.exit()

    siamesenet = retrieveTrainedSiameseModel(model_path, siamesenet)
    siamesenet.compile(optimizer='adam', loss=tf.losses.binary_crossentropy, metrics=[binaryAccuracy])

    if args.validation:

        print("VALIDATION")
        
        if args.French:
            print('French validation set loaded')
            couples_p_FR, couples_n_FR, _, _, sents_val_p_FR, sents_val_n_FR = get_couple_sent_combs(SUBS_FR, COUPLES_FR, 2, 4)    
            filename_val_p_FR = get_data_filename(couples_p_FR, sents_val_p_FR, 'P', 'solo')
            filename_val_p_FR = [DATA_PATH_FR + filename for filename in filename_val_p_FR]
            filename_val_n_FR = get_data_filename(couples_n_FR, sents_val_n_FR, 'N', 'solo')
            filename_val_n_FR = [DATA_PATH_FR + filename for filename in filename_val_n_FR]
            
            filename_val_pos = filename_val_p_FR
            filename_val_neg = filename_val_n_FR
            filename_val = filename_val_p_FR + filename_val_n_FR
            print(f'Validation set size: {len(filename_val)}')

        elif args.Italian:
            print('Italian validation set loaded')
            couples_p_ITA, couples_n_ITA, _, _, sents_val_p_ITA, sents_val_n_ITA = get_couple_sent_combs(SUBS_ITA, COUPLES_ITA, 2, 4)    
            filename_val_p_ITA = get_data_filename(couples_p_ITA, sents_val_p_ITA, 'P', 'solo')
            filename_val_p_ITA = [DATA_PATH_ITA + filename for filename in filename_val_p_ITA]
            filename_val_n_ITA = get_data_filename(couples_n_ITA, sents_val_n_ITA, 'N', 'solo')
            filename_val_n_ITA = [DATA_PATH_ITA + filename for filename in filename_val_n_ITA]
            
            filename_val_pos = filename_val_p_ITA
            filename_val_neg = filename_val_n_ITA
            filename_val = filename_val_p_ITA + filename_val_n_ITA
            print(f'Validation set size: {len(filename_val)}')

        elif args.All:
            couples_p_FR, couples_n_FR, _, _, sents_val_p_FR, sents_val_n_FR = get_couple_sent_combs(SUBS_FR, COUPLES_FR, 2, 4)    
            filename_val_p_FR = get_data_filename(couples_p_FR, sents_val_p_FR, 'P', 'solo')
            filename_val_p_FR = [DATA_PATH_FR + filename for filename in filename_val_p_FR]
            filename_val_n_FR = get_data_filename(couples_n_FR, sents_val_n_FR, 'N', 'solo')
            filename_val_n_FR = [DATA_PATH_FR + filename for filename in filename_val_n_FR]

            couples_p_ITA, couples_n_ITA, _, _, sents_val_p_ITA, sents_val_n_ITA = get_couple_sent_combs(SUBS_ITA, COUPLES_ITA, 2, 4)
            filename_val_p_ITA = get_data_filename(couples_p_ITA, sents_val_p_ITA, 'P', 'solo')
            filename_val_p_ITA = [DATA_PATH_ITA + filename for filename in filename_val_p_ITA]
            filename_val_n_ITA = get_data_filename(couples_n_ITA, sents_val_n_ITA, 'N', 'solo')
            filename_val_n_ITA = [DATA_PATH_ITA + filename for filename in filename_val_n_ITA]
        
        
            print('All validation set loaded')
            filename_val_pos = filename_val_p_FR + filename_val_p_ITA
            filename_val_neg = filename_val_n_FR + filename_val_n_ITA
            filename_val = filename_val_pos + filename_val_neg
            print(f'Validation set size: {len(filename_val)}')

        else:
            print('No validation set loaded, please use --validation flag and choose between --French, --Italian or --All')


        loss_val, acc_mean_val, acc_pos_val, acc_neg_val = model_eval(siamesenet, filename_val_pos, 
                                                                        filename_val_neg, batch=batchSize)

        sim_val_pos, sim_val_neg = plot_results(siamesenet,filename_val_pos,filename_val_neg, resPath, name='', set="VAL")
        
        test_val_loss.append(loss_val)
        test_val_acc.append(acc_mean_val)
        test_val_acc_pos.append(acc_pos_val)
        test_val_acc_neg.append(acc_neg_val)
        sim_pos_VAL.append(sim_val_pos)
        sim_neg_VAL.append(sim_val_neg)

        print("VALIDATION SET RESULTS")
        print('Loss (binary crossentropy)  is ' + str(loss_val))
        print('Accuracy is ' + str(acc_mean_val))
        print('Accuracy positive is ' + str(acc_pos_val))
        print('Accuracy negative is ' + str(acc_neg_val))
        print('\n')

        np.save(resPath + 'val_loss.npy', np.asarray(test_val_loss))
        np.save(resPath + 'val_acc.npy', np.asarray(test_val_acc))
        np.save(resPath + 'val_acc_pos.npy', np.asarray(test_val_acc_pos))
        np.save(resPath + 'val_acc_neg.npy', np.asarray(test_val_acc_neg))
        np.save(resPath + 'val_sim_pos.npy', np.asarray(sim_pos_VAL))
        np.save(resPath + 'val_sim_neg.npy', np.asarray(sim_neg_VAL))

    print("TEST")

    if args.same:

        print('Test mode: main0~main75 data, all negative labled')
        
        if args.French:
            print('French test set loaded')
            filename_test_pos = []
            filename_test_neg = glob.glob(PATH_TFR_FR + 'main0/N*') + glob.glob(PATH_TFR_FR + 'main25/N*') + glob.glob(PATH_TFR_FR + 'main50/N*') + glob.glob(PATH_TFR_FR + 'main75/N*') 
            filename_test = filename_test_pos + filename_test_neg
            print(f'Test set size: {len(filename_test)}')
        
        elif args.Italian:
            print('Italian test set loaded')
            filename_test_pos = []
            filename_test_neg = glob.glob(PATH_TFR_ITA + 'main0/N*') + glob.glob(PATH_TFR_ITA + 'main25/N*') + glob.glob(PATH_TFR_ITA + 'main50/N*') + glob.glob(PATH_TFR_ITA + 'main75/N*')
            filename_test = filename_test_pos + filename_test_neg
            print(f'Test set size: {len(filename_test)}')
        
        elif args.All:
            print('All test set loaded')
            filename_test_pos = []
            filename_test_neg = glob.glob(PATH_TFR_FR + 'main0/N*') + glob.glob(PATH_TFR_FR + 'main25/N*') + glob.glob(PATH_TFR_FR + 'main50/N*') + glob.glob(PATH_TFR_FR + 'main75/N*') + glob.glob(PATH_TFR_ITA + 'main0/N*') + glob.glob(PATH_TFR_ITA + 'main25/N*') + glob.glob(PATH_TFR_ITA + 'main50/N*') + glob.glob(PATH_TFR_ITA + 'main75/N*')
            filename_test = filename_test_pos + filename_test_neg
            print(f'Test set size: {len(filename_test)}')

        else:
            print('No test set loaded, choose between --french, --italian or --all')
            sys.exit()

        loss_test, acc_neg_test = model_eval_one_class(siamesenet, filename_test_neg, batch=batchSize)

        print("TEST SET RESULTS")
        print('Loss (binary crossentropy)  is ' + str(loss_test))
        print('Accuracy negative is ' + str(acc_neg_test))
        print('\n')

        end = time.time()
        print('TEST FINISHED, RUN TIME: ', end-start)
        sys.exit()


    if args.cross:

        print('Test mode: imitation data.')

        if args.French:
            print('French test set loaded')
            couples_p_FR, couples_n_FR, _, _, sents_test_p_FR, sents_test_n_FR = get_couple_sent_combs(SUBS_FR, COUPLES_FR, 2, 4)    
            filename_test_p_FR = get_data_filename(couples_p_FR, sents_test_p_FR, 'P', 'imitation')
            filename_test_p_FR = [DATA_TEST_FR + filename for filename in filename_test_p_FR]
            filename_test_n_FR = get_data_filename(couples_n_FR, sents_test_n_FR, 'N', 'imitation')
            filename_test_n_FR = [DATA_TEST_FR + filename for filename in filename_test_n_FR]

            filename_test_pos = filename_val_p_FR
            filename_test_neg = filename_val_n_FR
            filename_test = filename_test_pos + filename_test_neg
            print(f'Test set size: {len(filename_test)}')
        
        elif args.Italian:
            print('Italian test set loaded')
            couples_p_ITA, couples_n_ITA, _, _, sents_test_p_ITA, sents_test_n_ITA = get_couple_sent_combs(SUBS_ITA, COUPLES_ITA, 2, 4)
            filename_test_p_ITA = get_data_filename(couples_p_ITA, sents_test_p_ITA, 'P', 'imitation')
            filename_test_p_ITA = [DATA_TEST_ITA + filename for filename in filename_test_p_ITA]
            filename_test_n_ITA = get_data_filename(couples_n_ITA, sents_test_n_ITA, 'N', 'imitation')
            filename_test_n_ITA = [DATA_TEST_ITA + filename for filename in filename_test_n_ITA]

            filename_test_pos = filename_val_p_ITA
            filename_test_neg = filename_val_n_ITA
            filename_test = filename_test_pos + filename_test_neg
            print(f'Test set size: {len(filename_test)}')
        
        elif args.All:
            print('All test set loaded')
            couples_p_FR, couples_n_FR, _, _, sents_test_p_FR, sents_test_n_FR = get_couple_sent_combs(SUBS_FR, COUPLES_FR, 2, 4)    
            filename_test_p_FR = get_data_filename(couples_p_FR, sents_test_p_FR, 'P', 'imitation')
            filename_test_p_FR = [DATA_TEST_FR + filename for filename in filename_test_p_FR]
            filename_test_n_FR = get_data_filename(couples_n_FR, sents_test_n_FR, 'N', 'imitation')
            filename_test_n_FR = [DATA_TEST_FR + filename for filename in filename_test_n_FR]

            couples_p_ITA, couples_n_ITA, _, _, sents_test_p_ITA, sents_test_n_ITA = get_couple_sent_combs(SUBS_ITA, COUPLES_ITA, 2, 4)
            filename_test_p_ITA = get_data_filename(couples_p_ITA, sents_test_p_ITA, 'P', 'imitation')
            filename_test_p_ITA = [DATA_TEST_ITA + filename for filename in filename_test_p_ITA]
            filename_test_n_ITA = get_data_filename(couples_n_ITA, sents_test_n_ITA, 'N', 'imitation')
            filename_test_n_ITA = [DATA_TEST_ITA + filename for filename in filename_test_n_ITA]

            filename_test_pos = filename_val_p_FR + filename_val_p_ITA
            filename_test_neg = filename_val_n_FR + filename_val_n_ITA
            filename_test = filename_test_pos + filename_test_neg
            print(f'Test set size: {len(filename_test)}')

        else:
            print('No test set loaded, choose between --French, --Italian or --All')
            sys.exit()
        
    loss_test, acc_mean_test, acc_pos_test, acc_neg_test = model_eval(siamesenet, filename_test_pos, 
                                                                        filename_test_neg, batch=batchSize)

    sim_test_pos, sim_test_neg = plot_results(siamesenet,filename_test_pos,filename_test_neg, resPath, name='', set="TEST")

    test_result_loss.append(loss_test)
    test_result_acc.append(acc_mean_test)
    test_result_acc_pos.append(acc_pos_test)
    test_result_acc_neg.append(acc_neg_test)
    sim_pos_TEST.append(sim_test_pos)
    sim_neg_TEST.append(sim_test_neg)

    print("TEST SET RESULTS")
    print('Loss (binary crossentropy)  is ' + str(loss_test))
    print('Accuracy is ' + str(acc_mean_test))
    print('Accuracy positive is ' + str(acc_pos_test))
    print('Accuracy negative is ' + str(acc_neg_test))
    print('\n')

    np.save(resPath + 'test_loss.npy', np.asarray(test_result_loss))
    np.save(resPath + 'test_acc.npy', np.asarray(test_result_acc))
    np.save(resPath + 'test_acc_pos.npy', np.asarray(test_result_acc_pos))
    np.save(resPath + 'test_acc_neg.npy', np.asarray(test_result_acc_neg))
    np.save(resPath + 'test_sim_pos.npy', np.asarray(sim_pos_TEST))
    np.save(resPath + 'test_sim_neg.npy', np.asarray(sim_neg_TEST))
        
    end = time.time()
    print('TEST FINISHED, RUN TIME: ', end-start)
        
