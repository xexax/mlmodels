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
import numpy as np
import pandas as pd
import json
import importlib
from importlib import import_module
from pathlib import Path
from warnings import simplefilter
from datetime import datetime



####################################################################################################
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict, path_norm, params_json_load
from mlmodels.util import (get_recursive_files, load_config, log, os_package_root_path, path_norm)




####################################################################################################
def get_all_json_path(json_path):
    return get_recursive_files(json_path, ext='/*.json')


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
def metric_eval(actual=None, pred=None, metric_name="mean_absolute_error"):
	"""
      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html#sklearn.metrics.get_scorer 
	"""
    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
    return metric(actual, pred)

 
def preprocess_timeseries_m5(data_path=None, dataset_name=None, pred_length=10, item_id=None):
    df         = pd.read_csv(data_path + dataset_name)
    col_to_del = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    temp_df    = df.drop(columns=col_to_del).copy()
      
    # 1, -1 are hardcoded because we have to explicitly mentioned days column 
    temp_df    = pd.melt(temp_df, id_vars=["id"], value_vars=temp_df.columns[1: -1])

    # select one itemid for which we have to forecast
    i_id       = item_id
    temp_df    = temp_df.loc[temp_df["id"] == i_id]
    temp_df.rename(columns={"variable": "Day", "value": "Demand"}, inplace=True)
    # making df to compatible 3d shape, otherwise cannot be reshape to 3d compatible form
    pred_length = pred_length
    temp_df     = temp_df.iloc[:pred_length * (temp_df.shape[0] // pred_length)]
    temp_df.to_csv("{}/{}.csv".format(data_path, i_id), index=False)



####################################################################################################
def benchmark_run(bench_pars=None, args=None, config_mode="test"):
	"""
      Runnner for benchmark computation
      File is alredy saved on disk

	"""
    #pre_process(data_path=args.data_path, dataset_name=args.dataset_name, 
    #            pred_length=bench_pars["pred_length"], item_id=args.item_id)
      
    dataset_uri  = args.data_path + f"{args.item_id}.csv"
    json_path    = path_norm( args.path_json )
    output_path  = path_norm( args.path_out )
    json_list    = get_all_json_path(json_path)
    metric_list  = bench_pars['metric_list']
    benchmark_df = pd.DataFrame(columns=["date_run", "model_uri", "json",
                                         "dataset_uri", "metric", "metric_name"])

    for jsonf in json_list :
        log ( f"### Running {jsonf} #####")
        #### Model URI and Config JSON
        config_path = path_norm(jsonf)
        model_pars, data_pars, compute_pars, out_pars = params_json_load(config_path, config_mode= config_mode)


        #### Setup Model 
        model_uri = model_pars['model_uri']  # "model_tch.torchhub.py"
        module    = module_load(model_uri)
        model     = module.Model(model_pars, data_pars, compute_pars)
        
        #### Fit
        model, session = module.fit(model, data_pars, compute_pars, out_pars)          
    
        #### Inference Need return ypred, ytrue
        ypred, ytrue = module.predict(model=model, model_pars=model_pars, session=session, 
                                      data_pars=data_pars, compute_pars=compute_pars, 
                                      out_pars=out_pars, return_ytrue=1)   
        
        ### Calculate Metrics
        for ind, metric in enumerate(metric_list):
            metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
            benchmark_df.loc[ind, "date_run"]    = str(datetime.now())
            benchmark_df.loc[ind, "model_uri"]   = model_uri
            benchmark_df.loc[ind, "json"]        = str([model_pars, data_pars, compute_pars ])
            benchmark_df.loc[ind, "dataset_uri"] = dataset_uri
            benchmark_df.loc[ind, "metric_name"] = metric
            benchmark_df.loc[ind, "metric"]      = metric_val


    log( f"benchmark file saved at {output_path}")  
    os.makedirs( output_path, exist_ok=True)
    benchmark_df.to_csv( f"{output_path}/benchmark.csv", index=False)
    return benchmark_df





####################################################################################################
############CLI Command ############################################################################
def cli_load_arguments(config_file=None):
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    if config_file is None:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(cur_path, "config/benchmark_config.json")
    p = argparse.ArgumentParser()

    def add(*w, **kw):
        p.add_argument(*w, **kw)
    
    add("--config_file", default=config_file, help="Params File")
    add("--config_mode", default="test", help="test/ prod /uat")
    add("--log_file",    default="ztest/benchmark/mlmodels_log.log", help="log.log")
    add("--do",          default="run", help="do ")

    add("--data_path",   default="mlmodels/dataset/timeseries/", help="Dataset path")
    add("--dataset_name",default="sales_train_validation.csv", help="dataset name")   

    add("--path_json",   default="mlmodels/dataset/json/benchmark/", help="")

    ##### out pars
    add("--path_out",    default="ztest/benchmark/", help=".")



    add("--item_id",     default="HOBBIES_1_001_CA_1_validation", help="forecast for which item")

    arg = p.parse_args()
    return arg




def main():
    arg = cli_load_arguments()

    if arg.do == "timeseries":
        log("Fit")
        bench_pars = {"metric_list": ["mean_absolute_error", "mean_squared_error",
                                      "mean_squared_log_error", "median_absolute_error", 
                                      "r2_score"], 
                      "pred_length": 100}

        pre_process_timeseries_m5(data_path=arg.data_path, dataset_name=arg.dataset_name, 
                                  pred_length=bench_pars["pred_length"], item_id=arg.item_id)              
        benchmark_run(bench_pars, arg) 


if __name__ == "__main__":
    main()

