# coding: utf-8
"""
Generic template for new model.
Check parameters template in models_config.json

"model_pars":   { "learning_rate": 0.001, "num_layers": 1, "size": 6, "size_layer": 128, "output_size": 6, "timestep": 4, "epoch": 2 },
"data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] },
"compute_pars": { "distributed": "mpi", "epoch": 10 },
"out_pars":     { "out_path": "dataset/", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] }



"""
import inspect
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


VERBOSE = False
MODEL_URI = "model_XXXX/yyy.py"

####################################################################################################
######## Logs
from mlmodels.util import os_package_root_path, log
"""
def os_package_root_path(filepath, sublevel=0, path_add=""):

"""



####################################################################################################
class Model:
  def __init__(self, model_pars=None, data_pars=None, compute_pars=None  ):
    ### Model Structure        ################################
    self.model = None   #ex Keras model
    
    






def fit(model, data_pars={}, compute_pars={}, out_pars={},   **kw):
  """
  """

  sess = None # Session type for compute
  Xtrain, Xtest, ytrain, ytest = None, None, None, None  # data for training.
  

  return model, sess






def fit_metrics(model, data_pars={}, compute_pars={}, out_pars={},  **kw):
    """
       Return metrics of the model when fitted.
    """
    ddict = {}
    
    return ddict

    
        

  

def predict(model, sess=None, data_pars={}, compute_pars={}, out_pars={},  **kw):
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





def save(model=None, session=None, save_pars={}):
    from mlmodels.util import save_tf
    print(save_pars)
    save_tf(model, session, save_pars['path'])
     


def load(load_pars={}):
    from mlmodels.util import load_tf
    print(load_pars)
    input_tensors, output_tensors =  load_tf(load_pars['path'], 
                                            filename=load_pars['model_uri'])


    model = Model()
    model.model = None
    session = None
    return model, session




####################################################################################################
def get_dataset(data_pars=None, **kw):
  """
    JSON data_pars to get dataset
    "data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas",
    "size": [0, 0, 6], "output_size": [0, 6] },
  """

  if data_pars['train'] :
    Xtrain, Xtest, ytrain, ytest = None, None, None, None  # data for training.
    return Xtrain, Xtest, ytrain, ytest 


  else :
    Xtest, ytest = None, None  # data for training.
    return Xtest, ytest 



def path_setup(out_folder="ztest", sublevel=1, data_path="dataset/"):
    data_path = os_package_root_path(__file__, sublevel=sublevel, path_add=data_path)
    out_path = os.getcwd() + "/" + out_folder
    os.makedirs(out_path, exist_ok=True)

    model_path = out_path + "/model/"
    os.makedirs(model_path, exist_ok=True)

    log(data_path, out_path, model_path)
    return data_path, out_path, model_path


def get_params(param_pars={}, **kw):
    import json
    pp          = param_pars
    choice      = pp['choice']
    config_mode = pp['config_mode']
    data_path   = pp['data_path']

    if choice == "json":
       cf = json.load(open(data_path, mode='r'))
       cf = cf[config_mode]
       return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']


    if choice == "test01":
        log("#### Path params   ##########################################")
        data_path, out_path, model_path = path_setup(out_folder="/ztest/", sublevel=1,
                                                     data_path="dataset/")
        data_pars    = {}
        model_pars   = {}
        compute_pars = {}
        out_pars     = {}

        return model_pars, data_pars, compute_pars, out_pars

    else:
        raise Exception(f"Not support choice {choice} yet")




################################################################################################
########## Tests are normalized Do not Change ##################################################
from mlmodels.util import test_module


def test_api_global(model_uri="model_tf/1_lstm.py", data_path="dataset/", pars_choice="json", config_mode="test",
                    reset=True):
    ###loading the command line arguments
    # model_uri = "model_xxxx/yyyy.py"

    log("#### Loading params   ##############################################")
    param_pars = {"choice":pars_choice,  "data_path":data_path,  "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
    log(model_uri, model_pars, data_pars, compute_pars, out_pars)


    log("############ Model test Global  ###########################################")
    from mlmodels.models import test_api
    save_pars ={}
    test_api(model_uri, model_pars, data_pars, compute_pars, out_pars, save_pars)



def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice":pars_choice,  "data_path":data_path,  "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)


    log("#### Loading dataset   #############################################")
    xtuple = get_dataset(data_pars)


    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    model, session = fit(model, data_pars, compute_pars, out_pars)


    log("#### save the trained model  #######################################")
    save(model, session, out_pars)


    log("#### Predict   #####################################################")
    ypred = predict(model, session, data_pars, compute_pars, out_pars)


    log("#### metrics   #####################################################")
    metrics_val = fit_metrics(model, data_pars, compute_pars, out_pars)
    print(metrics_val)


    log("#### Plot   ########################################################")


    log("#### Save/Load   ###################################################")
    save(model, out_pars['modelpath'])
    model2 = load(out_pars['modelpath'])
    #     ypred = predict(model2, data_pars, compute_pars, out_pars)
    #     metrics_val = metrics(model2, ypred, data_pars, compute_pars, out_pars)
    print(model2)



if __name__ == '__main__':
    VERBOSE = True
    test_path = os.getcwd() + "/mytest/"


    ### Local
    test(pars_choice="json")
    test(pars_choice="test01")


    ### Global mlmodels
    test_api_global(model_uri="model_tf/1_lstm.py", data_path="dataset/", pars_choice="json", config_mode="test")


    ### module test
    test_module(model_uri="model_tf/1_lstm.py", data_path="dataset/", pars_choice="json", reset=True)




