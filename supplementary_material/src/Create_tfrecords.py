#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Aldo Pastore, Zheng Yuan
# Date created: 15/11/2021
# Date last modified: 02/06/2023
# Python Version: 3.9.13
# License: CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)

from re import S
import numpy as np
from itertools import combinations, product
import os
import tensorflow as tf
from Config import *
import argparse

#####Script Flags#############################################################################################
parser = argparse.ArgumentParser(description="Extract MFCCs from French or Italian wav files.")

data_mode = parser.add_mutually_exclusive_group()
data_mode.add_argument("-f", '--french', action="store_true", help="Choose the French audio dataset.")
data_mode.add_argument("-i", '--italian', action="store_true",  help="Choose the Italian audio dataset")

args = parser.parse_args()
##############################################################################################################


# get speaker combinations, with/without cross combination
def get_couple_combos(subs, cross=False):
    
    '''Get speaker combinations, with/without cross combination.
    
    :param: subs: list of speakers.
    :param: cross: if True, create dyadic combination for all speakers.
    
    '''
    if subs[0] % 2 == 0:
        couples_neg = np.asarray([[sub, sub + 1] for sub in subs if sub % 2 == 0])
    else:
        couples_neg = np.asarray([[sub, sub + 1] for sub in subs if sub % 2 == 1])
        
    couples_pos = np.asarray([[i, i] for i in subs])

    if cross:
        couples_neg = np.asarray(list(combinations(range(1, len(subs) + 1), 2)))

    return couples_pos, couples_neg


# get positive and negative sentence combinations
def get_sent_combos(nSent, sess, pos=True):

    '''Get positive and negative sentence combinations.
        
        :param: nSent: number of sentences.
        :param: sess: name of sessions.
        :param: pos: if True, create sentence combinations for the same speaker.
        
        '''

    # different speakers
    if sess in ['solo', 'imitation']:
        if pos:
            sent_combos = np.asarray(list(combinations(range(1, nSent + 1), 2)))
            return sent_combos
        sent_combos = np.asarray(list(product(range(1, nSent + 1), range(1, nSent + 1))))
        return sent_combos

    else:    
        # sess in ['main0', 'main25', 'main50', 'main75']
        sents1 = [str(ele)  for ele in range(1, nSent +1) if ele % 2 == 1]
        sents2 = [str(ele)  for ele in range(1, nSent +1) if ele % 2 == 0]
        
        if sess in ['main0', 'main75']:
            if pos:
                return None
            sent_combos = np.asarray(list(zip(sents1, sents2)))
            return sent_combos

        else:
            if pos:
                return None
            sent_combos = np.asarray(list(zip(sents2, sents1)))
            return sent_combos


    


def serialize_sequence(audio_sequence1, audio_sequence2, labels):

    '''Serialize sequence to tfrecord.

    :param: audio_sequence1: audio sequence vector of sentence 1.
    :param: audio_sequence2: audio sequence vector of sentence 2.
    :param: labels: list of data labels.
    
    '''
    # The object we return
    ex = tf.train.SequenceExample()

    # A non-sequential feature of our example
    seq1_length = len(audio_sequence1)
    seq2_length = len(audio_sequence2)
    ex.context.feature["feat1_length"].int64_list.value.append(seq1_length)
    ex.context.feature["feat2_length"].int64_list.value.append(seq2_length)

    fl_audio1_feat = ex.feature_lists.feature_list["audio1_feat"]
    fl_audio2_feat = ex.feature_lists.feature_list["audio2_feat"]
    fl_audio_labels = ex.feature_lists.feature_list["labels"]

    for audio_feat in audio_sequence1:
        fl_audio1_feat.feature.add().float_list.value.extend(audio_feat)
    for audio_feat in audio_sequence2:
        fl_audio2_feat.feature.add().float_list.value.extend(audio_feat)
    for label in labels:
        fl_audio_labels.feature.add().int64_list.value.append(label)

    return ex


def fnorm(data, normtype='m0s1'):

    '''
    Perform data normalization.

    :param: data: a numpy ndarray like object.
    :param: normtype: choose from maxmin normalization 'minmax' or z-score normalization 'm0s1'.
    
    '''

    if normtype == 'minmax':
        M = np.max(data, 0)
        m = np.min(data, 0)
        ndata = (data - m) / (M - m)

        return ndata

    else:
        m = np.mean(data, 0)
        sd = np.std(data, 0)
        ndata = (data - m) / sd

        return ndata


def build_couple_tfrecords(load_path, save_path, sub1, sub2, sent1, sent2, name, sess):

    '''
    Build sentence pair tfrecords file.

    :param: load_path: str--path to load MFCC data.
    :param: save_path: str--path to save tfrecords data.
    :param: sub1: int--speaker 1 id.
    :param: sub2: int--speaker 2 id.
    :param: sent1: int--sentence 1 id.
    :param: sent2: int--sentence 2 id.
    :param: name: str--positive or negative label of the tfrecord file.
    :param: sess1: str--session 1 name.
    :param: sess2: str--session 2 name.

    '''

    label = np.asarray((sub1 == sub2), dtype='int64')
    
    p1 = load_path + str(sub1) + "_" + sess + "_" + str(sent1) + ".npy"
    print(p1)

    p2 = load_path + str(sub2) + "_" + sess + "_" + str(sent2) + ".npy"
    print(p2)

    if os.path.exists(p1) and os.path.exists(p2):
        wave1 = np.load(p1)
        wave2 = np.load(p2)

        norm_wave1 = fnorm(wave1.T)
        norm_wave2 = fnorm(wave2.T)

        outfilename = name + "-s" + str(sub1) + "_sent" + str(sent1) + '_' + sess + "-s" + str(sub2) + "_sent" + str(
            sent2) + '_' + sess + ".tfrecords"

        fp = open(save_path + outfilename, 'w')
        writer = tf.io.TFRecordWriter(fp.name)

        # serialize example
        label = np.array([label]).astype('int64')
        serialized_sentence = serialize_sequence(norm_wave1, norm_wave2, label)

        # write to tfrecord
        writer.write(serialized_sentence.SerializeToString())
        writer.close()
        fp.close()
        return True

    else:
        print("path not exists")
        return False


def createTFR_couples(pLoad, pSave, subs, sess):

    '''
    Create tfrecords file for a couple of speakers.

    :param: pLoad: str--path to load MFCC data.
    :param: pSave: str--path to save tfrecords data.
    :param: subs: list--list of speaker ids.
    :param: sess1: str--session name.

    '''

    couples_pos, couples_neg = get_couple_combos(subs)
    count_pos = 0

    ## BUILDING POSITIVE DATA SAMPLES (SAME SPEAKERS)  ##
    for couple in couples_pos:
        
        # make save path
        pSaveP = pSave +  sess
        if not os.path.isdir(pSaveP):
            os.mkdir(pSaveP)
        pSaveP = pSaveP + "/"

        if sess in ['solo', 'imitation']:
            sent_comb_pos = get_sent_combos(SESSIONS[sess], sess, pos=True)
            for sents in sent_comb_pos:
                built = build_couple_tfrecords(load_path=pLoad, save_path=pSaveP, sub1=couple[0], sub2=couple[1],
                                            sent1=sents[0], sent2=sents[1], name="P",
                                            sess=sess)
                if built:
                    print("sub" + str(couple[0]) + " sent" + str(sents[0]) + ' ' + sess
                        + " AND sub" + str(couple[1]) + " sent" + str(sents[1]) + ' ' + sess)
                    count_pos += 1

    print(f'{count_pos} positive tfrecord files built.')

    # BUILDING NEGATIVE DATA SAMPLES (DIFFERENT SPEAKERS)

    count_neg = 0

    for couple in couples_neg:
        
        # make save path
        pSaveN = pSave + sess
        if not os.path.isdir(pSaveN):
            os.mkdir(pSaveN)
        pSaveN = pSaveN + "/"

        sent_comb_neg = get_sent_combos(SESSIONS[sess], sess, pos=False)
        for sents in sent_comb_neg:
            built = build_couple_tfrecords(load_path=pLoad, save_path=pSaveN, sub1=couple[0], sub2=couple[1],
                                        sent1=sents[0], sent2=sents[1], name="N",
                                        sess=sess)
                
            if built:
                print("sub" + str(couple[0]) + " sent" + str(sents[0]) + ' ' + sess
                        + " AND sub" + str(couple[1]) + " sent" + str(sents[1]) + ' ' + sess)

                count_neg += 1

    print(f'{count_neg} negative tfrecord files built.')

########################################################################################################################
if __name__ == "__main__":

    print('START CREATING TFR')
    if args.french:
        for sess in SESSIONS.keys():
            print(f'Creating TFR for {sess}')
            createTFR_couples(pLoad=PATH_LOAD_FR, pSave=PATH_TFR_FR, subs=SUBS_FR, sess=sess)
    if args.italian:
        for sess in SESSIONS.keys():
            print(f'Creating TFR for {sess}')
            createTFR_couples(pLoad=PATH_LOAD_ITA, pSave=PATH_TFR_ITA, subs=SUBS_ITA, sess=sess)
    
    print('FINISHED')

