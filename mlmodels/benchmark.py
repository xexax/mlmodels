# -*- coding: utf-8 -*-
"""

For all json in Json_path_list :
   Load_json, 
   Load_model, 
   run_model,
   get_metrics, 
   add_to_dataframe



"""
import argparse
import glob
import inspect
import json
import os
import re
import sys
from importlib import import_module
from pathlib import Path
from warnings import simplefilter

####################################################################################################
from mlmodels.util import (get_recursive_files, load_config, log, os_package_root_path, path_norm)


def get_all_json_path(json_path):
    return get_recursive_files(json_path, ext='/*.json')


####################################################################################################
def run_benchmark_all(bench_pars=None, json_path=None config_mode="test"):
    from mlmodels.models import module_load
    from mlmodels.util import path_norm_dict, path_norm, params_json_load
    import json
    import pandas as pd
    from datetime import datetime

    json_list = get_all_json_path(json_path)
    metric_list = bench_pars['metric_list']
    benchmark_df = pd.DataFrame(columns=["date_run", "model_uri", "json",  
                                        "dataset_uri", "metric", "metric_name"])
    for jsonf in json_list :
        log ("### Running {} #####".format(jsonf))
        #### Model URI and Config JSON
        config_path = path_norm(jsonf)
        config_mode = config_mode

        #### PLEASE Include   model_pars['model_uri'] = "model_gluon.gluon_prophet.py"  in the JSON directly
        model_pars, data_pars, compute_pars = params_json_load(config_path, config_mode= config_mode)
         
        """ 
        #### Model Parameters
        if "prophet" in jsonf:
            model_pars, data_pars, compute_pars = \
            params_json_load(config_path, config_mode= config_mode)
            model_pars['model_uri'] = "model_gluon.gluon_prophet.py"
            
        elif "deepar" in jsonf:
            model_pars, data_pars, compute_pars = \
            params_json_load(config_path, config_mode= config_mode)
            model_pars['model_uri'] = "model_gluon.gluon_deepar.py"
        else:
            model_pars, data_pars, compute_pars, out_pars = \
            params_json_load(config_path, config_mode= config_mode)
            
        model_uri = model_pars['model_uri']  # "model_tch.torchhub.py"
        """
      
      
        #### Setup Model 
        model_uri = model_pars['model_uri']  # "model_tch.torchhub.py" 
        module = module_load(model_uri)
        model = module.Model(model_pars, data_pars, compute_pars) 
        
        #### Fit
        model, session = module.fit(model, data_pars, compute_pars, out_pars)           #### fit model
        
        

        #### Inference  Please change to return ypred, ytrue
        ypred, ytrue = module.predict(model=model, model_pars=model_pars, session=session, 
                               data_pars=data_pars, compute_pars=compute_pars, 
                               out_pars=out_pars, return_ytrue=1)   
        


        ### Calculate Metrics
        for ind, metric in enumerate(metric_list):
            """https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html#sklearn.metrics.get_scorer
            
              
            
            """
            from sklearn.metrics import get_scorer
            scorer_metric = get_scorer(metric)
            
            metrics_val = scorer_metric(ytrue, ypred)
            
            
            ##### NO Metrics IS INDEPENDANT Of model 
            """
            metrics_val = module.fit_metrics(model, data_pars, compute_pars, 
                                            out_pars, model_pars, wmape=metric)   #### Check fit metrics
            """
            
            
            benchmark_df.loc[ind, "date_run"] = str(datetime.now())
            benchmark_df.loc[ind, "model_uri"] = model_uri
            benchmark_df.loc[ind, "json"] = jsonf
            benchmark_df.loc[ind, "dataset_uri"] = "~/dataset/timeseries/HOBBIES_1_001_CA_1_validation.csv"
            benchmark_df.loc[ind, "metric_name"] = metric
            benchmark_df.loc[ind, "metric"] = metrics_val["wmape"]

    log(benchmark_df)
    ##### Output Format :
    """
     Dataframe :
         [ "date_run", model_uri", "json",  "dataset_uri",  "metric", "metric_name" ]


    """





####################################################################################################
def config_model_list(folder=None):
    # Get all the model.py into folder
    folder = os_package_root_path() if folder is None else folder
    # print(folder)
    module_names = get_recursive_files(folder, r'/*model*/*.py')
    mlist = []
    for t in module_names:
        mlist.append(t.replace(folder, "").replace("\\", "."))
        print(mlist[-1])

    return mlist




####################################################################################################
############CLI Command ############################################################################
def cli_load_arguments(config_file=None):
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    if config_file is None:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(cur_path, "config/benchmark_config.json")
    # print(config_file)

    p = argparse.ArgumentParser()

    def add(*w, **kw):
        p.add_argument(*w, **kw)

    add("--config_file", default=config_file, help="Params File")
    add("--config_mode", default="test", help="test/ prod /uat")
    add("--log_file",    default="ztest/benchmark/mlmodels_log.log", help="log.log")

    add("--do",          default="run", help="do ")


    add("--path_json",   default="dataset/json/benchmark/", help="")


    ##### out pars
    add("--path_out",    default="ztest/benchmark/", help=".")

    arg = p.parse_args()
    # arg = load_config(arg, arg.config_file, arg.config_mode, verbose=0)
    return arg


def main():
    arg = cli_load_arguments()
    if arg.do == "run":
        log("Fit")
        bench_pars = {"metric_list": ["wmape"]}
        run_benchmark_all(bench_pars, arg.path_json) 


if __name__ == "__main__":
    main()



