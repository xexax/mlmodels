# coding: utf-8
"""
Generic template for new model.
Check parameters template in models_config.json

"model_pars":   { "learning_rate": 0.001, "num_layers": 1, "size": 6, "size_layer": 128, "output_size": 6, "timestep": 4, "epoch": 2 },
"data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] },
"compute_pars": { "distributed": "mpi", "epoch": 10 },
"out_pars":     { "out_path": "dataset/", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] }



"""
import os, sys, inspect
from datetime import datetime, timedelta

import numpy as np
import pandas as pd




####################################################################################################
def os_module_path():
  current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  parent_dir = os.path.dirname(current_dir)
  # sys.path.insert(0, parent_dir)
  return parent_dir


def os_file_path(data_path):
  from pathlib import Path
  data_path = os.path.join(Path(__file__).parent.parent.absolute(), data_path)
  print(data_path)
  return data_path


def os_package_root_path(filepath, sublevel=0, path_add=""):
  """
     get the module package root folder
  """
  from pathlib import Path
  path = Path(filepath).parent
  for i in range(1, sublevel + 1):
    path = path.parent
  
  path = os.path.join(path.absolute(), path_add)
  return path
# print("check", os_package_root_path(__file__, sublevel=1) )


def log(*s, n=0, m=1):
  sspace = "#" * n
  sjump = "\n" * m
  print(sjump, sspace, s, sspace, flush=True)


class to_namespace(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

  def get(self, key):
    return self.__dict__.get(key)



####################################################################################################
class Model:
  def __init__(self, model_pars=None, data_pars=None
               ):
    ### Model Structure        ################################
    self.model = None
    
    





def fit(model, data_pars={}, compute_pars={}, out_pars={}, out_pars={},  **kwargs):
  """

  :param model:    Class model
  :param data_pars:  dict of
  :param out_pars:
  :param compute_pars:
  :param kwargs:
  :return:
  """

  sess = None # Session type for compute
  Xtrain, Xtest, ytrain, ytest = None, None, None, None  # data for training.
  
  
  

  return model, sess


def metrics(ytrue, ypred, yproba=None, model=None, sess=None, data_pars={}, out_pars={}, **kw):
    """
       Return metrics 
    """
    ddict = {}
    
    
    return ddict
  
  

def predict(model, sess=None, data_pars=None, out_pars=None, compute_pars=None, **kw):
  ##### Get Data ###############################################
  Xpred, ypred = None, None

  #### Do prediction
  ypred = model.model.fit(Xpred)

  ### Save Results
  
  
  ### Return val
  if compute_pars.get("return_pred_not") is not None :
    return ypred


  
  
def reset_model():
  pass



####################################################################################################
def get_dataset(choice="", data_pars=None, **kw):
  """
    JSON data_pars to get dataset
    "data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas",
    "size": [0, 0, 6], "output_size": [0, 6] },
  """
  d = to_namespace(data_pars)
  print(d)
  df = None

  if d.data_type == "pandas" :
    df = pd.DataFrame(d.data_path)
  
  #######################################
  return df



def get_params(choice="test", data_path="", config_mode="test",  **kwargs):
  # Get sample parameters of the model

  if choice == "json":
     return {}
  
  if choice == "test":
    p = {"learning_rate": 0.001, "num_layers": 1, "size": None, "size_layer": 128,
         "output_size": None, "timestep": 4, "epoch": 2,}
    









if __name__ == "__main__":
  test()
  
  
  
  
  
  
  
  
