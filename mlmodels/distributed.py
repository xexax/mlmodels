# -*- coding: utf-8 -*-
"""
Distributed




 

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
from mlmodels.util import path_norm_dict,  params_json_load
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

  

def create_conda_env() :
    pass
 
 
def process_launch(uri="mlmodels.utils:myFun", args={}, **kw):
    """
      Execute a function on a new python env and get back results in the current env.

    """
    
    path      = uri_get_path(uri)
    deps_file = uri_get_depsfile(uri)   ## deps files from the folder
    env_name  = uri_get_envname(uri)    # conda env name

    ### Meta Script which package the Execution.
    script    = uri_get_script(uri)
    iid       = random.randomint

    if env_name is None :
      env_name = uri_create_envname(uri)
      bash_script = f"""
       
       conda create -n {env_name} python=3.6.9 -y  && \
       pip install -r {deps_file}  && \
       source activate {env_name} &&  \
       python {script}

       """

    else :
      bash_script = f"""       
       source activate {env_name} &&  \
       python {script} {iid}

      """


    bash_script_file = f"tmp/mybash_script_{iid}.sh"
    with open(bash_script_file, mode="w") as f :
      f.writelines(bash_script)

        
    #### Sync Launch or Asynchronous launch
    cmd = " chmod 777 .  && {bash_script_file} "
    myid = subprocess.run(  cmd  )
 
 
    while not finished :
       sleep(10)
       
    ddict = read_from_disk( myid, iid)
    return ddict
    

def uri_get_script():    
    x = """
      script.py
         from mlmodels.utilss import myfun
       
         args = read_from_disk(myid)
       
         res = myfun(**args)
       
        write_on_disk(myid, res)
    """



       
       


"""
    ### my values = EXecute( "Myfunction" , args  ) in a separate python interpreter....
    
    
 
 def create_conda_env()
 
 
 def process_launch(uri="mlmodels.utils:myFun", args={}, **kw):
    
    write_on_disk(myid, args)
    
    create_or_find_conda_env()
    
    bash_script ="
       source activate myenv
       python mymain.py
    "


    
    #### Sync Launch or Asynchronous launch
    myid = subprocess.launch(   cmd )
 
 
    while not finished :
       sleep(10)
       
    ddict = read_from_disk(myid)
    return ddict
    
    
    
    mymain.py
       from mlmodels.utilss import myfun
       
       args = read_from_disk(myid)
       
       res = myfun(**args)
       
       write_on_disk(myid, res)
       
       
      
 
 
 """
   



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
    
    add("--config_file"    , default=config_file                        , help="Params File")
    add("--config_mode"    , default="test"                             , help="test/ prod /uat")
    add("--log_file"       , default="ztest/benchmark/mlmodels_log.log" , help="log.log")

    add("--do"             , default="fit"             , help="do ")


    add("--n_node"             , default=2           , help="do ")
    add("--model_uri"          , default="model_tch.mlp"           , help="do ")
    add("--model_json"        , default=""           , help="do ")


    arg = p.parse_args()
    return arg



def main():
    arg = cli_load_arguments()
    """

    """ 
    import mlmodels
    log(arg.do)
    path = mlmodels.path[0]

    if arg.do == "fit":

       n_node     = arg.n_node
       model_uri  = arg.model_uri
       model_json = arg.model_json

       cmd = f"python  {path}/distri_torch_mpirun.sh   {n_node}    {model_uri}    {model_json}" 
       print(cmd)
       os.system( cmd )


    else :
        raise Exception("No options")



if __name__ == "__main__":
    main()




