from __future__ import print_function

import os
import sys


def import_data_tch(name="", mode="train", node_id=0, data_folder_root=""):
    import torch.utils.data.distributed
    from torchvision import datasets, transforms

    if name == "mnist" :
        data_folder = os.path.join( data_folder_root,  "data-%d" % node_id)
        dataset = datasets.MNIST(
            data_folder,
            train=True if mode =="train" else False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        return dataset


def import_data_fromfile(**kw):
   """
       data_pars["data_path"]
   
   """ 
   import pandas as pd
   import numpy as np

   m = kw
   if m.get("uri_type") in ["pickle", "pandas_pickle" ]:
       df = pd.read_pickle(m["data_path"])
       return df


   if m.get("uri_type") in ["csv", "pandas_csv" ]:
       df = pd.read_csv(m["data_path"])
       return df


   if m.get("uri_type") in [ "dask" ]:
       df = pd.read_csv(m["data_path"])
       return df


   if ".npz" in m['data_path'] :
       arr = import_to_numpy(data_pars, mode=mode, **kw)
       return arr


   if ".csv" in m['data_path'] or ".txt" in m['data_path']  :
       df = import_to_pandas(data_pars, mode=mode, **kw)
       return df


   if ".pkl" in m['data_path']   :
       df = pd.read_pickle(m["data_path"], **kw)
       return df






def import_data_dask(**kw):
  extension = kw['path'].split(".")[-1]

  if kw.get("use_dask", False):
    import dask.dataframe as dd
    if extension in [".csv", ".txt"]: 
       df = dd.read_csv(kw["data_path"])
    elif extension in [".pkl"]: 
       df = dd.read_pickle(kw["data_path"])
    elif extension in [".npz"]: 
       df = dd.read_pickle(m["data_path"])
    else: raise Exception(f"Not support extension {extension}")
  return df









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





def get_dataset(data_pars) : 
  """
    path:
    is_local  : Local to the repo 
    data_type:
    train : 1/0
    data_source :  ams
  """
  dd = data_pars

  if not d.get('is_local') is None :
      dd['path'] = os_package_root_path(__file__, sublevel=0, dd['path'] )


  if dd['train'] :
     df = pd.read_csv(path) 



     ### Donwload from external


     ## Get from csv, local 


     ## Get from csv, external


     ### Get from external tool


