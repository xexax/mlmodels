# -*- coding: utf-8 -*-
"""



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
from mlmodels.util import (get_recursive_files, load_config, log, os_package_root_path)



####################################################################################################
def run_benchmark_all(bench_pars=None, json_path):
    from mlmodels.models import module_load
    from mlmodels.util import path_norm_dict, path_norm, params_json_load
    import json

    json_list = get_all_json_path(json_path)

    for jsonf in json_list :
        #### Model URI and Config JSON
        config_path = path_norm( jsonf  ) # 'model_tch/torchhub_cnn.json'
        config_mode = "test"  ### test/prod


        #### Model Parameters
        hypermodel_pars, model_pars, data_pars, compute_pars, out_pars = params_json_load(config_path, config_mode= config_mode)
        print( hypermodel_pars, model_pars, data_pars, compute_pars, out_pars)
        model_uri   = model_pars['model_uri']  # "model_tch.torchhub.py"


        #### Setup Model 
        module         = module_load( model_uri)
        model          = module.Model(model_pars, data_pars, compute_pars) 
        
        #### Fit
        model, session = module.fit(model, data_pars, compute_pars, out_pars)           #### fit model
        metrics_val    = module.fit_metrics(model, data_pars, compute_pars, out_pars)   #### Check fit metrics
        print(metrics_val)


        #### Inference
        ypred          = module.predict(model, session, data_pars, compute_pars, out_pars)   
        print(ypred)



        ### Calculate Metrics


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
    add("--log_file", default="ztest/benchmark/mlmodels_log.log", help="log.log")

    add("--do", default="run", help="do ")


    add("--path_json", default="dataset/json/benchmark/", help="")


    ##### out pars
    add("--path_out", default="ztest/benchmark/", help=".")

    arg = p.parse_args()
    # arg = load_config(arg, arg.config_file, arg.config_mode, verbose=0)
    return arg


def main():
    arg = cli_load_arguments()
    print(arg.do)

    if arg.do == "run":
        log("Fit")
        bench_pars = None
        run_benchmark_all(bench_pars, arg.path_json) 


if __name__ == "__main__":
    main()
