# -*- coding: utf-8 -*-
"""
Pipeline :


https://www.neuraxio.com/en/blog/neuraxle/2019/10/26/neat-machine-learning-pipelines.html
https://github.com/Neuraxio/Neuraxle


>>> from sklearn.compose import ColumnTransformer
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> from sklearn.preprocessing import OneHotEncoder
>>> column_trans = ColumnTransformer(
...     [('city_category', OneHotEncoder(dtype='int'),['city']),
...      ('title_bow', CountVectorizer(), 'title')],
...     remainder='drop')

>>> column_trans.fit(X)
ColumnTransformer(transformers=[('city_category', OneHotEncoder(dtype='int'),
                                 ['city']),
                                ('title_bow', CountVectorizer(), 'title')])

>>> column_trans.get_feature_names()
['city_category__x0_London', 'city_category__x0_Paris', 'city_category__x0_Sallisaw',
'title_bow__bow', 'title_bow__feast', 'title_bow__grapes', 'title_bow__his',
'title_bow__how', 'title_bow__last', 'title_bow__learned', 'title_bow__moveable',
'title_bow__of', 'title_bow__the', 'title_bow__trick', 'title_bow__watson',
'title_bow__wrath']


"""

import os

import pandas as pd
from sklearn.decomposition import TruncatedSVD


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


####################################################################################################
def pd_na_values(df, cols=None, default=0.0, **kw):
    cols = cols if cols is not None else list(df.columns)
    for t in cols:
        df[t] = df[t].fillna(default)

    return df


def TruncatedSVD_fun(df, n_components, **kw):
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    return svd.fit_transform(df.values)


def onehot_encoder(df, cols=None, **kw):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    cols = cols if cols is not None else list(df.columns)
    for idx, col in enumerate(cols):
        df[col] = le.fit_transform(df[col])
    return df


###################################################################################################
def pd_concat(df1, df2, colid1):
    df3 = df1.join(df2.set_index(colid), on=colid, how="left")
    return df3


def pipe_split(in_pars, out_pars, compute_pars, **kw):
    df = pd.read_csv(in_pars['in_path'])
    colid = in_pars['colid']
    path = out_pars['out_path']
    file_list = {}

    for colname, cols in in_pars['col_group'].items():
        dfi = df[[colid] + cols].set_index(colid)
        os.makedirs(f"{path}/{colname}/", exist_ok=True)
        fname = f'{path}/{colname}/df_{colname}.pkl'

        dfi.to_pickle(fname)
        log(colname, fname, cols)
        file_list[colname] = fname

    return file_list


def pipe_merge(in_pars, out_pars, compute_pars=None, **kw):
    dfall = None
    for filename in in_pars['file_list']:
        log(filename)
        dfi = pd.read_pickle(filename)
        dfall = df if dfall is None else pd_concat(dfall, dfi, in_pars['colid'])

    dfall.to_pickle(out_pars['out_path'])
    return dfall


def pipe_load(df, **in_pars):
    path = in_pars['in_path']
    log(path)

    if ".pkl" in path:
        df = pd.read_pickle(path)

    elif path[-4:] in ['.csv', '.txt', 'gz']:
        df = pd.read_csv(path)

    else:
        return None

    log("file loaded", df.head(3))
    return df


def pipe_run_fit(pipe_list, in_pars, out_pars, compute_pars=None, checkpoint=True, **kw):
    """
      Save the processsor state
    :return:
    """
    log('Start execution')
    dfin = None
    for (pname, pexec, args) in pipe_list:
        out_file = out_pars['out_path'] + f"/{pname}/dfout.pkl"
        log(pname, pexec, out_file)
        os.makedirs(out_pars['out_path'] + f"/{pname}/", exist_ok=True)

        #######
        if args.get("saved_model"):
            pexec_ = load_model(args.get("saved_model"))


        elif args.get("model_class"):
            ##### Class approach
            pexec_ = pexec(**args)
        else:
            #### Functional approach
            # dfout = pexec(dfin, **args)
            from sklearn.preprocessing import FunctionTransformer
            pexec_ = FunctionTransformer(pexec, kw_args=args, validate=False)

        pexec_.fit(dfin)
        dfout = pexec_.transform(dfin)

        dfin = dfout
        if checkpoint:
            pipe_checkpoint(dfout, {'out_path': out_file, 'type': 'pandas'})
            pipe_checkpoint(pexec_, {'out_path': out_file, 'type': 'model'})

    return dfout


def pipe_run_inference(pipe_list, in_pars, out_pars, compute_pars=None, checkpoint=True, **kw):
    """
    :Only using the processing, no saving
    :return:
    """
    log('Start execution')
    dfin = None
    for (pname, pexec, args) in pipe_list:
        out_file = out_pars['out_path'] + f"/{pname}/dfout.pkl"
        log(pname, pexec, out_file)
        os.makedirs(out_pars['out_path'] + f"/{pname}/", exist_ok=True)

        #######
        if args.get("saved_model"):
            pexec_ = load_model(args.get("saved_model"))

        pexec_.fit(dfin)
        dfout = pexec_.transform(dfin)

        dfin = dfout
        if checkpoint:
            pipe_checkpoint(dfout, {'out_path': out_file, 'type': 'pandas'})

    return dfout


def pipe_checkpoint(df, out_path, **kw):
    if kw.get("saved_model"):   df.to_pickle(out_path)


def load_model(path):
    return pickle.load(open(path, mode='b'))


def save_model(model, path):
    pickle.save(model, open(path, mode='b'))


###################################################################################################
def test(data_path="/dataset/", pars_choice="json"):
    ### Local test
    root = os_package_root_path(__file__, 0)
    out_path = f"{os.getcwd()}/ztest/pipeline_01/"
    log("#### Loading params   ##############################################")
    in_pars = {"in_path": f"{root}/{data_path}/movielens_sample.txt",
               "colid": "user_id",
               "col_group": {"colnum": ["rating", "movie_id", "age"],
                             "colcat": ["genres", "gender"]}

               }

    out_pars = {"out_path": out_path
                }

    compute_pars = {"cpu": True}

    ### Split data
    file_list = pipe_split(in_pars, out_pars, compute_pars)

    ### Pipeline colnum
    in_pars['in_path'] = file_list['colnum']
    pipe_list = [("00_Load_data", pipe_load, in_pars),
                 ("01_NA_values", pd_na_values, {"default": 0.0, "out_path": out_path}),
                 ("02_SVD", TruncatedSVD_fun, {"n_components": 1, "out_path": out_path}),
                 ("03_save", pipe_checkpoint, {"out_path": out_path}),
                 ]
    pipe_run_fit(pipe_list, in_pars, out_pars, compute_pars)

    ### Pipeline colcat
    in_pars['in_path'] = file_list['colcat']
    pipe_list = [("00_Load_data", pipe_load, in_pars),
                 ("01_NA_values", pd_na_values, {"default": 0.0, "out_path": out_path}),
                 ("011_onehot_encoder", onehot_encoder, {"cols": ["genres", "gender"]}),
                 ("02_SVD", TruncatedSVD_fun, {"n_components": 1, "out_path": out_path}),
                 ]
    pipe_run_fit(pipe_list, in_pars, out_pars, compute_pars)

    log("#### save the trained model  #######################################")
    # save(model, data_pars["modelpath"])


    log("#### metrics   ####################################################")

    log("#### Plot   #######################################################")

    log("#### Save/Load   ##################################################")
    # print(model2)


if __name__ == '__main__':
    VERBOSE = True
    test(pars_choice="json")

