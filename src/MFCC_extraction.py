#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Aldo Pastore, Zheng Yuan
# Date created: 15/11/2021
# Date last modified: 02/06/2023
# Python Version: 3.9.13
# License: CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)


from pydoc import splitdoc
import librosa
import os
import numpy as np
import tensorflow as tf
import argparse


#####Script Flags#############################################################################################
parser = argparse.ArgumentParser(description="Extract MFCCs from French or Italian wav files.")

data_mode = parser.add_mutually_exclusive_group()
data_mode.add_argument("-f", '--french', action="store_true", help="Choose the French audio dataset.")
data_mode.add_argument("-i", '--italian', action="store_true",  help="Choose the Italian audio dataset")

args = parser.parse_args()
##############################################################################################################


def computeMFCC(path_to_wav, wav_name, sr=44100, frameWidth=0.025, frameShift=0.010, nMFCC=13, mono=True, delta=False,
                deltadelta=False):
    '''
    Compute MFCCs from wav file using librosa.
    
    :param path_to_wav: path to wav files
    :param wav_name: name of wav files
    :param sr: sampling rate
    :param frameWidth: frame width in seconds
    :param frameShift: frame shift in seconds
    :param nMFCC: number of MFCCs
    :param mono: if True, convert to mono
    :param delta: if True, compute delta MFCCs
    :param deltadelta: if True, compute delta delta MFCCs
    :return: MFCCs
    
    '''

    y, _ = librosa.core.load(path_to_wav + wav_name, sr, mono=mono)
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=nMFCC, hop_length=int(frameShift * sr), n_fft=int(frameWidth * sr))
    out = mfcc

    # computing deltas (if specified)
    if delta:
        d = librosa.feature.delta(mfcc, width=5)
        out = np.concatenate((out, d), axis=0)

        if deltadelta:
            dd = librosa.feature.delta(d, width=5)
            out = np.concatenate((out, dd), axis=0)
    saver_name = wav_name.replace('wav', 'npy')
    
    # saving MFCCs
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if deltadelta:
        saver = SAVE_PATH + 'mfcc_dd/'
    else:
        if delta:
            saver = SAVE_PATH + 'mfcc_d/'
        else:
            saver = SAVE_PATH + 'mfcc/'

    if not os.path.exists(saver):
        os.makedirs(saver)

    saver_name = change_MFCC_name(saver_name)
    np.save(saver + saver_name, out)

    return out, saver + saver_name

def change_MFCC_name(filename):

    '''
    Change the name of the MFCCs to match the name of the tfrecord files.
    e.g. d0102_2_solo555_1_mono.npy -> 2_solo_1.npy

    :param path: path to MFCCs.
    '''
    oldname = filename
    split = oldname.split('_')
    if split[2] in ['solo555', 'imitation555']:
        split[2] = split[2][:-3]

    newname = str(split[1] + '_' + split[2] + '_' + split[3] + '.npy')
    print(oldname, '->', newname)

    return newname

##############################################################################################################
if __name__ == "__main__":
    
    if args.french:
        LOAD_PATH = '../data/FR/wav/'
        SAVE_PATH = '../data/FR/'
    
    elif args.italian:
        LOAD_PATH = '../data/ITA/wav/'
        SAVE_PATH = '../data/ITA/'

    wav_list = [file for file in os.listdir(LOAD_PATH) if file.endswith('.wav')]
    mfcc_list = []

    MaxL = 0
    # Each MFCC is [Nfeatures x Nframes] array
    for wav_file in wav_list:
        print('creating MFCC for:  ' + wav_file)
        mfcc, mfcc_path = computeMFCC(LOAD_PATH, wav_file, delta=True, deltadelta=True)
        mfcc_list.append(mfcc_path)

        if mfcc.shape[-1] > MaxL: MaxL = mfcc.shape[-1]


    # MFCC PADDING
    for mfcc_path in mfcc_list:
        mfcc = np.load(mfcc_path, allow_pickle=True)
        mfcc_padded = tf.keras.preprocessing.sequence.pad_sequences(mfcc, padding='pre', value=0.0, maxlen=MaxL,
                                                                    dtype='float32')
        np.save(mfcc_path, mfcc_padded)

    print(f'{len(mfcc_list)} MFCC files created.')
