# -*- coding: utf-8 -*-
import copy
import math
import os
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import scipy as sci

import sklearn as sk
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer



####################################################################################################
import tensorflow as tf
import torch as torch
import autogluon
import gluonts




def main():
  print("os.getcwd", os.getcwd())
  print(np, np.__version__) 
  print(tf, tf.__version__)
  print(torch, torch.__version__)

<<<<<<< HEAD
  test_list =[
  "python model_gluon/gluon_automl.py",
  "python model_gluon/gluon_deepar.py",
  "python model_gluon/gluon_ffn.py",
=======

  test_list =[
   "python model_gluon/gluon_automl.py",
   "python model_gluon/gluon_deepar.py",
   "python model_gluon/gluon_ffn.py",
>>>>>>> 71314ca4b49d0968e1243bc7dfff3497899d861b

  ###
  "python model_keras/01_deepctr.py",


  ###
  "python model_keras/01_deepctr.py",


  ###
  "python model_tf/1_lstm.py",


  ###
  "python model_tch/nbeats.py",
  ]

  ]

  for cmd in test_list :
    print(cmd)
    os.system( cmd )




if __name__ == "__main__":
    main()





