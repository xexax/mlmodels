# -*- coding: utf-8 -*-
"""
 ml_test --do test_benchmark

 
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
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict, path_norm, params_json_load
from datetime import datetime

####################################################################################################
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict,  params_json_load
from mlmodels.util import (get_recursive_files, load_config, log, os_package_root_path, path_norm)


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
    pred[pred < 0] = 0
    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
    return metric(actual, pred)


def preprocess_timeseries_m5(data_path=None, dataset_name=None, pred_length=10, item_id=None):
    df = pd.read_csv(data_path + dataset_name)
    cal = pd.read_csv(data_path + "calendar.csv")
    col_to_del = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    temp_df = df.drop(columns=col_to_del).copy()
    temp_df = pd.melt(temp_df, id_vars=["id"], value_vars=temp_df.columns[1: -1])
    # select one itemid for which we have to forecast
    i_id = "HOBBIES_1_001_CA_1_validation"
    temp_df = temp_df.loc[temp_df["id"] == i_id]
    temp_df.rename(columns={"variable": "Day", "value": "Demand"}, inplace=True)
    temp_df["Date"] = temp_df["Day"].map(cal.set_index("d")["date"].to_dict())
    temp_df.set_index("Date", drop=True, inplace=True)
    temp_df.drop(columns=["id", "Day"], inplace=True)
    # making df to compatible 3d shape, otherwise cannot be reshape to 3d compatible form
    time_steps = 60
    temp_df = temp_df.iloc[:time_steps * (temp_df.shape[0] // time_steps)]
    train = temp_df.iloc[:-time_steps]
    test = temp_df.iloc[-time_steps:]
    train.to_csv("{}/{}_train.csv".format(data_path, i_id))
    test.to_csv("{}/{}_test.csv".format(data_path, i_id))


####################################################################################################
def benchmark_run(bench_pars=None, args=None, config_mode="test"):
    preprocess_timeseries_m5(data_path=args.data_path, dataset_name=args.dataset_name,
                             pred_length=bench_pars["pred_length"], item_id=args.item_id)

    dataset_uri = args.data_path + "{}.csv".format(args.item_id)
    json_path = args.path_json
    output_path = args.path_out
    json_list = get_all_json_path(json_path)
    metric_list = bench_pars['metric_list']
    benchmark_df = pd.DataFrame(columns=["date_run", "model_uri", "json",
                                         "dataset_uri", "metric_name", "metric"])
    df_ind = 0
    for ind, jsonf in enumerate(json_list):
        log("### Running {} #####".format(jsonf))
        # Model URI and Config JSON
        config_path = path_norm(jsonf)
        config_mode = config_mode

        # PLEASE Include
        model_pars, data_pars, compute_pars, out_pars = params_json_load(
            config_path, config_mode=config_mode)

        # Setup Model
        model_uri = model_pars['model_uri']  # "model_tch.torchhub.py"
        module = module_load(model_uri)
        model = module.Model(model_pars, data_pars, compute_pars)
        try:
            # Fit
            model, session = module.fit(model, data_pars, compute_pars, out_pars)  # fit model
        except Exception as e:
            # Fit
            model = model.model
            model, session = module.fit(model, data_pars, compute_pars, out_pars)  # fit model

        # Inference  Please change to return ypred, ytrue
        ypred, ytrue = module.predict(model=model, model_pars=model_pars, session=session,
                                      data_pars=data_pars, compute_pars=compute_pars,
                                      out_pars=out_pars, return_ytrue=1)

        actual = np.array(ytrue).reshape(-1, 1)
        pred = np.array(ypred).reshape(-1, 1)
        # Calculate Metrics
        for metric in metric_list:
            """https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html#sklearn.metrics.get_scorer 
            """
            metric_val = metric_eval(actual=actual, pred=pred,
                                     metric_name=metric)
            benchmark_df.loc[df_ind, "date_run"] = str(datetime.now())
            benchmark_df.loc[df_ind, "model_uri"] = model_uri
            benchmark_df.loc[df_ind, "json"] = jsonf
            # i hardcoded data path because there will be only dataset for benchmark
            benchmark_df.loc[df_ind, "dataset_uri"] = dataset_uri
            benchmark_df.loc[df_ind, "metric_name"] = metric
            benchmark_df.loc[df_ind, "metric"] = metric_val
            df_ind += 1

    os.makedirs(path_norm(output_path), exist_ok=True)
    log(" benchmark file saved at {}".format(path_norm(output_path)))
    benchmark_df.to_csv("{}benchmark.csv".format(path_norm(output_path)), index=False)
    return benchmark_df
    # Output Format :
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
    p = argparse.ArgumentParser()

    def add(*w, **kw):
        p.add_argument(*w, **kw)
    add("--config_file", default=config_file, help="Params File")
    add("--config_mode", default="test", help="test/ prod /uat")
    add("--log_file", default="ztest/benchmark/mlmodels_log.log", help="log.log")
    add("--data_path", default="mlmodels/dataset/timeseries/", help="Dataset path")
    add("--dataset_name", default="sales_train_validation.csv", help="dataset name")
    add("--do", default="timeseries", help="do ")
    add("--item_id", default="HOBBIES_1_001_CA_1_validation", help="forecast for which item")
    add("--path_json", default="mlmodels/dataset/json/benchmark/", help="")
    # out pars
    add("--path_out", default="mlmodels/ztest/benchmark/", help=".")

    arg = p.parse_args()
    # arg = load_config(arg, arg.config_file, arg.config_mode, verbose=0)
    return arg


def main():
    arg = cli_load_arguments()
    print("[DEBUG] from main arg {}\n".format(arg))
    if arg.do == "timeseries":
        log("Time series model")
        bench_pars = {"metric_list": ["mean_absolute_error", "mean_squared_error",
                                      "mean_squared_log_error", "median_absolute_error",
                                      "r2_score"],
                      "pred_length": 100}

                      
        #### Pre-processing
        arg.data_path = "dataset/timeseries/"
        arg.dataset_name = "sales_train_validation.csv"
        preprocess_timeseries_m5(data_path = arg.data_path, 
                                 dataset_name = arg.dataset_name, 
                                 pred_length = bench_pars["pred_length"], item_id=arg.item_id)   

        #### Models
        arg.path_json = "dataset/json/benchmark_timeseries/"
        arg.path_out =  "ztest/benchmark_timseries/"

        benchmark_run(bench_pars, arg) 



    elif arg.do == "vision_mnist":
        log("Vision models")

        arg.data_path = ""
        arg.dataset_name = ""
        arg.path_json = "dataset/json/benchmark_cnn/"
        arg.path_out =  "ztest/benchmark_cnn/"

        bench_pars = {"metric_list": ["accuracy_score"]}
        benchmark_run(bench_pars=bench_pars, args=arg)




    else :
        raise Exception("No options")


if __name__ == "__main__":
    main()







# Benchmarking for CNN on MNIST #
def benchmark_run_mnist(bench_pars=None, args=None, config_mode="test"):
    """
      Runnner for benchmark computation on MNIST
      File is alredy saved on disk
	"""
    json_path    = path_norm( args.path_json )
    output_path  = path_norm( args.path_out )
    json_list    = get_all_json_path(json_path)
    metric_list  = bench_pars['metric_list']
    benchmark_df = pd.DataFrame(columns=["date_run", "model_uri", "json",
                                         "dataset_uri", "metric", "metric_name"])

    print(json_path)
    # import pdb; pdb.set_trace()
    for jsonf in json_list :
        log ( f"### Running {jsonf} #####")
        config_path = path_norm(jsonf)
        model_pars, data_pars, compute_pars, out_pars = params_json_load(config_path, config_mode= config_mode)

        log("#### Model init   #############################################")
        # import pdb; pdb.set_trace()
        model_uri = model_pars['model_uri']  
        module    = module_load(model_uri)
        model     = module.Model(model_pars, data_pars, compute_pars)


        log("#### Model fit   #############################################")
        model, session = module.fit(model, data_pars, compute_pars, out_pars)

        #### Inference Need return ypred, ytrue
        ypred, ytrue = module.predict(model=model, session=session, 
                                      data_pars=data_pars, compute_pars=compute_pars, 
                                      out_pars=out_pars, return_ytrue=1) 


        
        ### Calculate Metrics
        for ind, metric in enumerate(metric_list):
            metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
            benchmark_df.loc[ind, "date_run"]    = str(datetime.now())
            benchmark_df.loc[ind, "model_uri"]   = model_uri
            benchmark_df.loc[ind, "json"]        = str([model_pars, data_pars, compute_pars ])
            # benchmark_df.loc[ind, "dataset_uri"] = dataset_uri
            benchmark_df.loc[ind, "metric_name"] = metric
            benchmark_df.loc[ind, "metric"]      = metric_val

        # import pdb; pdb.set_trace()

    log( f"benchmark file saved at {output_path}")  
    os.makedirs( output_path, exist_ok=True)
    benchmark_df.to_csv( f"{output_path}/benchmark_MNIST_CNN.csv", index=False)
    return benchmark_df

