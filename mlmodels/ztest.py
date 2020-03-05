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


import mlmodels



def main():
  print("os.getcwd", os.getcwd())
  print(np, np.__version__) 
  print(tf, tf.__version__)
  print(torch, torch.__version__)


  path = mlmodels.__path__[0]
  

  test_list =[
   ###
   f"python {path}/model_tf/1_lstm.py",
 
   ###
   f"python {path}/model_keras/01_deepctr.py",

   ###
   f"python {path}/model_tch/nbeats.py",


   ###
   f"python {path}/model_gluon/gluon_deepar.py",
   f"python {path}/model_glufon/gluon_ffn.py",

    
   #### Too slow
   # f"python {path}/model_gluon/gluon_automl.py",

  ]


  for cmd in test_list :
    print("\n\n\n", flush=True)
    print(cmd, flush=True)
    os.system( cmd )




if __name__ == "__main__":
    main()





