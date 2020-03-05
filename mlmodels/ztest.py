# -*- coding: utf-8 -*-
import copy
import math
import os
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import scipy as sci
import sklearn as sk

####################################################################################################
import tensorflow as tf
import torch as torch
import autogluon
import gluonts



####################################################################################################
import mlmodels


####################################################################################################
def get_recursive_files(folderPath, ext='/*model*/*.py'):
  import glob
  files = glob.glob( folderPath + ext, recursive=True) 
  return files


def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)



####################################################################################################
def os_package_root_path(add_path="",n=0):
  from pathlib import Path
  add_path = os.path.join(Path(__file__).parent.absolute(), add_path)
  # print("os_package_root_path,check", add_path)
  return add_path


def os_package_root_path(filepath, sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    path = Path(os.path.realpath(filepath)).parent
    for i in range(1, sublevel + 1):
        path = path.parent

    path = os.path.join(path.absolute(), path_add)
    return path

  
def os_file_current_path():
  val = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  # return current_dir + "/"
  # Path of current file
  # from pathlib import Path

  # val = Path().absolute()
  val = str(os.path.join(val, ""))
  # print(val)
  return val



def model_get_list(folder=None, block_list=[]):
  # Get all the model.py into folder  
  folder = os_package_root_path() if folder is None else folder
  # print(folder)
  module_names = get_recursive_files(folder, r'/*model*//*model*/*.py' )                       


  NO_LIST = [  "__init__.py", "util", "preprocess" ]
  NO_LIST = NO_LIST + block_list


  for t in module_names :
      t = t.replace(folder, "").replace("\\", ".")

      flag = False     
      for x in NO_LIST :
        if x in t: FLAG = True

      if not flag  :
       list_select.append( t )
 


def main():
  print("os.getcwd", os.getcwd())
  print(np, np.__version__) 
  print(tf, tf.__version__)
  print(torch, torch.__version__)
  print(mlmodels) 

  path = mlmodels.__path__[0]
  

  model_list = model_get_list(folder=None, block_list=[])
  print(model_list)


  test_list =[    
   ### Tflow

   f"python {path}/model_tf/1_lstm.py",
 
    
   ### Keras
   f"python {path}/model_keras/01_deepctr.py",

    
   ### Torch
   f"python {path}/model_tch/nbeats.py",


   ###
   f"python {path}/model_gluon/gluon_deepar.py",
   f"python {path}/model_glufon/gluon_ffn.py",


   #### Too slow
   # f"python {path}/model_gluon/gluon_automl.py",


  ]

  
  for cmd in test_list :
    print(cmd, flush=True)
    os.system( cmd )




if __name__ == "__main__":
    main()





