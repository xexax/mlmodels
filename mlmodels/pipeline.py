# -*- coding: utf-8 -*-
"""

Pipeline :


https://www.neuraxio.com/en/blog/neuraxle/2019/10/26/neat-machine-learning-pipelines.html


https://github.com/Neuraxio/Neuraxle



"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np



####################################################################################################
# Helper functions
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


def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)





def pipe_split(in_pars, out_pars, compute_pars, **kw) :

    df = pd.read_csv(in_pars['in_path'])
    colid = in_pars['colid']
    for colname, cols in out_pars['col_group'] :
       dfi =  df[ [colid] + cols ]
       dfi.to_pickle( out_pars['out_path']  + f'{colname}' )
       log(colname, cols )


def pipe_merge(in_pars, out_pars, compute_pars, **kw) :
    pass



def pipe_load(in_pars) :
    df = pd.read_csv(in_pars['in_path'])
    return df


def pipeline_run( pipe_list, in_pars, out_pars, compute_pars, **kw) :
    """



    :param pipe_list:
    :return:
    """
    dfin = pipe_load(in_pars)
    for (pname, pexec, args) in pipe_list :
        try :
          log(pname, pexec, out_pars['out_path']+  "/{pname}/dfout.pkl" )
          dfout = pexec(dfin, **args)
          dfout.to_pickle( out_pars['out_path'] +  "/{pname}/dfout.pkl"  )
          dfin = dfout
        except Exception as e :
          log(pname, e)

    return dfout

"""

pipe_split(in_pars, out_pars, compute_pars, **kw) 

pipeline_run( pipe_list, in_pars={ df_colcat }, out_pars, compute_pars, **kw) 




"""

