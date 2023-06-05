#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Aldo Pastore, Zheng Yuan
# Date created: 15/11/2021
# Date last modified: 02/06/2023
# Python Version: 3.9.13
# License: CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)

import os

ROOT = os.path.abspath('../')

PATH_LOAD_FR = ROOT + '/data/FR/mfcc_dd/'
PATH_LOAD_ITA = ROOT + '/data/ITA/mfcc_dd/'

PATH_SAVE_FR = ROOT + '/data/FR/'
PATH_SAVE_ITA = ROOT + '/data/ITA/'

PATH_TFR_FR = ROOT + '/data/FR/tfrecords/'
PATH_TFR_ITA = ROOT + '/data/ITA/tfrecords/'

PATH_MODEL_FR = ROOT + '/models/FR/checkpoints/'
PATH_MODEL_ITA = ROOT + '/models/ITA/checkpoints/'
PATH_MODEL_FR_ITA = ROOT + '/models/FR_ITA/checkpoints/'
PATH_MODEL_VCTK_FR_ITA = ROOT + '/models/VCTK_FR_ITA/checkpoints/'

PATH_SAVE_MODEL = ROOT + '/models/'
PATH_SAVE_RESULTS = ROOT + '/results/'

DATA_PATH_FR = PATH_TFR_FR + 'solo/'
DATA_PATH_ITA = PATH_TFR_ITA + 'solo/'
DATA_TEST_FR = PATH_TFR_FR + 'imitation/'
DATA_TEST_ITA = PATH_TFR_ITA + 'imitation/'


COUPLES_FR = [['42', '43']]
COUPLES_ITA = [['1', '2']]
SUBS_FR = [42, 43]
SUBS_ITA = [1, 2]

SESSIONS = {"solo": 4, "main0": 4, "main25": 4, "main50": 4, "main75": 4, "imitation": 4}
