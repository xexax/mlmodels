# -*- coding: utf-8 -*-
import os
import re

# import toml
from pathlib import Path
import json


"""
from mlmodels.util import (os_package_root_path, log, path_norm
    config_load_root, config_path_pretrained, config_path_dataset, config_set
    )
"""



def metrics_evaluate(metric_list=["mean_squared_error"], ytrue=None, ypred=None, ypred_proba=None):

    """

##### Classification metrics

accuracy_score(y_true, y_pred,...)	Accuracy classification score.
auc(x, y)	Compute Area Under the Curve (AUC) using the trapezoidal rule
average_precision_score(y_true, y_score)	Compute average precision (AP) from prediction scores
balanced_accuracy_score(y_true, y_pred)	Compute the balanced accuracy
brier_score_loss(y_true, y_prob,...)	Compute the Brier score.
classification_report(y_true, y_pred)	Build a text report showing the main classification metrics
cohen_kappa_score(y1, y2,...)	Cohenâ€™s kappa: a statistic that measures inter-annotator agreement.
confusion_matrix(y_true, y_pred,...)	Compute confusion matrix to evaluate the accuracy of a classification.
dcg_score(y_true, y_score , k, ...  )	Compute Discounted Cumulative Gain.
f1_score(y_true, y_pred,...)	Compute the F1 score, also known as balanced F-score or F-measure
fbeta_score(y_true, y_pred, beta,...)	Compute the F-beta score
hamming_loss(y_true, y_pred,...)	Compute the average Hamming loss.
hinge_loss(y_true, pred_decision,...)	Average hinge loss (non-regularized)
jaccard_score(y_true, y_pred,...)	Jaccard similarity coefficient score
log_loss(y_true, y_pred , eps, ...  )	Log loss, aka logistic loss or cross-entropy loss.
matthews_corrcoef(y_true, y_pred,...)	Compute the Matthews correlation coefficient (MCC)
multilabel_confusion_matrix(y_true, ...)	Compute a confusion matrix for each class or sample
ndcg_score(y_true, y_score , k, ...  )	Compute Normalized Discounted Cumulative Gain.
precision_recall_curve(y_true, ...)	Compute precision-recall pairs for different probability thresholds
precision_recall_fscore_support(...)	Compute precision, recall, F-measure and support for each class
precision_score(y_true, y_pred,...)	Compute the precision
recall_score(y_true, y_pred,...)	Compute the recall
roc_auc_score(y_true, y_score,...)	Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
roc_curve(y_true, y_score,...)	Compute Receiver operating characteristic (ROC)
zero_one_loss(y_true, y_pred,...)	Zero-one classification loss.



##### Regression metrics

explained_variance_score(y_true, y_pred)	Explained variance regression score function
max_error(y_true, y_pred)	max_error metric calculates the maximum residual error.
mean_absolute_error(y_true, y_pred)	Mean absolute error regression loss
mean_squared_error(y_true, y_pred,...)	Mean squared error regression loss
mean_squared_log_error(y_true, y_pred)	Mean squared logarithmic error regression loss
median_absolute_error(y_true, y_pred)	Median absolute error regression loss
r2_score(y_true, y_pred,...)	R^2 (coefficient of determination) regression score function.
mean_poisson_deviance(y_true, y_pred)	Mean Poisson deviance regression loss.
mean_gamma_deviance(y_true, y_pred)	Mean Gamma deviance regression loss.
mean_tweedie_deviance(y_true, y_pred)	Mean Tweedie deviance regression loss.



##### Multilabel ranking metrics

coverage_error(y_true, y_score,...)	Coverage error measure
label_ranking_average_precision_score(...)	Compute ranking-based average precision
label_ranking_loss(y_true, y_score)	Compute Ranking loss measure



##### Clustering metrics

supervised, which uses a ground truth class values for each sample.
unsupervised, which does not and measures the â€˜qualityâ€™ of the model itself.

adjusted_mutual_info_score(...,...)	Adjusted Mutual Information between two clusterings.
adjusted_rand_score(labels_true, ...)	Rand index adjusted for chance.
calinski_harabasz_score(X, labels)	Compute the Calinski and Harabasz score.
davies_bouldin_score(X, labels)	Computes the Davies-Bouldin score.
completeness_score(labels_true, ...)	Completeness metric of a cluster labeling given a ground truth.
cluster.contingency_matrix(...,...)	Build a contingency matrix describing the relationship between labels.
fowlkes_mallows_score(labels_true, ...)	Measure the similarity of two clusterings of a set of points.
homogeneity_completeness_v_measure(...)	Compute the homogeneity and completeness and V-Measure scores at once.
homogeneity_score(labels_true, ...)	Homogeneity metric of a cluster labeling given a ground truth.
mutual_info_score(labels_true, ...)	Mutual Information between two clusterings.
normalized_mutual_info_score(...,...)	Normalized Mutual Information between two clusterings.
silhouette_score(X, labels,...)	Compute the mean Silhouette Coefficient of all samples.
silhouette_samples(X, labels , metric  )	Compute the Silhouette Coefficient for each sample.
v_measure_score(labels_true, labels_pred)	V-measure cluster labeling given a ground truth.


##### Biclustering metrics

consensus_score(a, b , similarity  )	The similarity of two sets of biclusters.



##### Pairwise metrics

pairwise.additive_chi2_kernel(X , Y  )	Computes the additive chi-squared kernel between observations in X and Y
pairwise.chi2_kernel(X , Y, gamma  )	Computes the exponential chi-squared kernel X and Y.
pairwise.cosine_similarity(X , Y, ...  )	Compute cosine similarity between samples in X and Y.
pairwise.cosine_distances(X , Y  )	Compute cosine distance between samples in X and Y.
pairwise.distance_metrics()	Valid metrics for pairwise_distances.
pairwise.euclidean_distances(X , Y, ...  )	Considering the rows of X (and Y=X) as vectors, compute the distance matrix between each pair of vectors.
pairwise.haversine_distances(X , Y  )	Compute the Haversine distance between samples in X and Y
pairwise.kernel_metrics()	Valid metrics for pairwise_kernels
pairwise.laplacian_kernel(X , Y, gamma  )	Compute the laplacian kernel between X and Y.
pairwise.linear_kernel(X , Y, ...  )	Compute the linear kernel between X and Y.
pairwise.manhattan_distances(X , Y, ...  )	Compute the L1 distances between the vectors in X and Y.
pairwise.nan_euclidean_distances(X)	Calculate the euclidean distances in the presence of missing values.
pairwise.pairwise_kernels(X , Y, ...  )	Compute the kernel between arrays X and optional array Y.
pairwise.polynomial_kernel(X , Y, ...  )	Compute the polynomial kernel between X and Y.
pairwise.rbf_kernel(X , Y, gamma  )	Compute the rbf (gaussian) kernel between X and Y.
pairwise.sigmoid_kernel(X , Y, ...  )	Compute the sigmoid kernel between X and Y.
pairwise.paired_euclidean_distances(X, Y)	Computes the paired euclidean distances between X and Y
pairwise.paired_manhattan_distances(X, Y)	Compute the L1 distances between the vectors in X and Y.
pairwise.paired_cosine_distances(X, Y)	Computes the paired cosine distances between X and Y
pairwise.paired_distances(X, Y , metric  )	Computes the paired distances between X and Y.
pairwise_distances(X , Y, metric, ...  )	Compute the distance matrix from a vector array X and optional Y.
pairwise_distances_argmin(X, Y,...)	Compute minimum distances between one point and a set of points.
pairwise_distances_argmin_min(X, Y)	Compute minimum distances between one point and a set of points.
pairwise_distances_chunked(X , Y, ...  )	Generate a distance matrix chunk by chunk with optional reduction

    
    """
    import pandas as pd
    from sklearn.metrics import get_scorer
    mdict = {  "metric_name": [],
               "metric_val":  [],
               "n_sample" :   [len(ytrue)] * len(metric_list)}

    for metric_name in metric_list :
      metric_scorer = get_scorer(metric_name)
    
      if ypred_proba is not None :
         mval = metric_scorer(ytrue, ypred, ypred_proba)
      else :
         mval = metric_scorer(ytrue, ypred)

      mdict["metric_name"].append(metric_name)
      mdict["metric_val"].append(mval)

    mdict = pd.DataFrame(mdict)
    return mdict



####################################################################################################
class to_namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

    def get(self, key):
        return self.__dict__.get(key)


def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)


####################################################################################################
def os_folder_copy(src, dst):
    """Copy a directory structure overwriting existing files"""
    import shutil
    for root, dirs, files in os.walk(src):
        if not os.path.isdir(root):
            os.makedirs(root)

        for file in files:
            rel_path = root.replace(src, '').lstrip(os.sep)
            dest_path = os.path.join(dst, rel_path)

            if not os.path.isdir(dest_path):
                os.makedirs(dest_path)

            shutil.copyfile(os.path.join(root, file), os.path.join(dest_path, file))



def os_get_file(folder=None, block_list=[], pattern=r'*.py'):
    # Get all the model.py into folder
    folder = os_package_root_path() if folder is None else folder
    # print(folder)
    module_names = get_recursive_files3(folder, pattern)
    print(module_names)


    NO_LIST = []
    NO_LIST = NO_LIST + block_list

    list_select = []
    for t in module_names:
        t = t.replace(folder, "").replace("\\", ".").replace(".py", "")

        flag = False
        for x in NO_LIST:
            if x in t: flag = True

        if not flag:
            list_select.append(t)

    return list_select


def model_get_list(folder=None, block_list=[]):
    # Get all the model.py into folder
    folder = os_package_root_path(__file__) if folder is None else folder
    # print(folder)
    module_names = get_recursive_files(folder, r'/*model*/*.py')

    NO_LIST = ["__init__", "util", "preprocess"]
    NO_LIST = NO_LIST + block_list

    list_select = []
    for t in module_names:
        t = t.replace(folder, "").replace("\\", ".")

        flag = False
        for x in NO_LIST:
            if x in t: flag = True

        if not flag:
            list_select.append(t)

    return list_select


def get_recursive_files2(folderPath, ext):
    results = os.listdir(folderPath)
    outFiles = []
    for file in results:
        if os.path.isdir(os.path.join(folderPath, file)):
            outFiles += get_recursive_files(os.path.join(folderPath, file), ext)
        elif re.match(ext, file):
            outFiles.append( folderPath + "/" + file)

    return outFiles


def get_recursive_files3(folderPath, ext):
    results = os.listdir(folderPath)
    outFiles = []
    for file in results:
        if os.path.isdir(os.path.join(folderPath, file)):
            outFiles += get_recursive_files(os.path.join(folderPath, file), ext)
        elif re.match(ext, file):
            outFiles.append(file)
    return outFiles



def get_model_uri(file):
  return Path(os.path.abspath(file)).parent.name + "." + os.path.basename(file).replace(".py", "")



def get_recursive_files(folderPath, ext='/*model*/*.py'):
    import glob
    files = glob.glob(folderPath + ext, recursive=True)
    return files





def path_norm(path=""):
    root = os_package_root_path(__file__, 0)

    if len(path) == 0 or path is None:
        path = root

    tag_list = [ "model_", "//model_",  "dataset", "template", "ztest", "example"  ]


    for t in tag_list :
        if path.startswith(t) :
            path = os.path.join(root, path)
            return path
    return path


def path_norm_dict(ddict):
    for k,v in ddict.items():
        if "path" in k :
            ddict[k] = path_norm(v)
    return ddict


"""
def os_module_path():
    import inspect
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    # sys.path.insert(0, parent_dir)
    return parent_dir
"""

def os_package_root_path(filepath="", sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    import mlmodels, os, inspect 

    path = Path(inspect.getfile(mlmodels)).parent
    # print( path )

    # path = Path(os.path.realpath(filepath)).parent
    for i in range(1, sublevel + 1):
        path = path.parent

    path = os.path.join(path.absolute(), path_add)
    return path


def os_file_current_path():
    import inspect
    val = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # return current_dir + "/"
    # Path of current file
    # from pathlib import Path

    # val = Path().absolute()
    val = str(os.path.join(val, ""))
    # print(val)
    return val


####################################################################################################
def test_module(model_uri="model_tf/1_lstm.py", data_path="dataset/", pars_choice="json", reset=True):
    ###loading the command line arguments
    # model_uri = "model_xxxx/yyyy.py"

    log("#### Module init   #################################################")
    from mlmodels.models import module_load
    module = module_load(model_uri)
    log(module)

    log("#### Loading params   ##############################################")
    param_pars = {"choice": pars_choice, "data_path": data_path}
    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)

    log("#### Run module test   ##############################################")
    from mlmodels.models import test_module as test_module_global
    test_module_global(model_uri, model_pars, data_pars, compute_pars, out_pars)


####################################################################################################
def config_load_root():
    import json
    path_user = os.path.expanduser('~')
    path_config = path_user + "/.mlmodels/config.json"

    ddict = json.load(open(path_config, mode='r'))
    return ddict


def config_path_pretrained():
    ddict = config_load_root()
    return ddict['model_trained']


def config_path_dataset():
    ddict = config_load_root()
    return ddict['dataset']


def config_set(ddict2):
    ddict = config_load_root()

    for k,x in ddict2.items():
      ddict[k] = x

    json.dump(ddict, open(ddict, mode='w'))
   



def params_json_load(path, config_mode="test"):
    import json
    pars = json.load(open(path, mode="r"))
    pars = pars[config_mode]

    ### HyperParam, model_pars, data_pars,
    list_pars = []
    for t in ["hypermodel_pars", "model_pars", "data_pars", "compute_pars", "out_pars"]:
        pdict = pars.get(t)
        if pdict:
            list_pars.append(pdict)
        else:
            log("error in json, cannot load ", t)

    return tuple(list_pars)



def load_config(args, config_file, config_mode, verbose=0):
    ##### Load file dict_pars as dict namespace #############################
    import json
    print(config_file) if verbose else None

    try:
        pars = json.load(open(config_file, mode="r"))
        # print(arg.param_file, model_pars)

        pars = pars[config_mode]  # test / prod
        print(config_file, pars) if verbose else None

        ### Overwrite dict_pars from CLI input and merge with toml file
        for key, x in vars(args).items():
            if x is not None:  # only values NOT set by CLI
                pars[key] = x

        # print(model_pars)
        pars = to_namespace(pars)  # like object/namespace model_pars.instance
        return pars

    except Exception as e:
        print(e)
        return args


def val(x, xdefault):
    try:
        return x if x is not None else xdefault
    except:
        return xdefault


####################################################################################################
####################################################################################################
def env_conda_build(env_pars=None):
    if env_pars is None:
        env_pars = {'name': "test", 'python_version': '3.6.5'}

    p = env_pars
    cmd = f"conda create -n {p['name']}  python={p['python_version']}  -y"
    print(cmd)
    os.system(cmd)


def env_pip_requirement(env_pars=None):
    from time import sleep
    if env_pars is None:
        env_pars = {'name': "test", 'requirement': 'requirements.txt'}

    root_path = os_package_root_path(__file__)
    p = env_pars
    # cmd = f"source activate {p['name']}  &&  "
    cmd = ""
    cmd = cmd + f"  pip install -r  {root_path}/{p['requirement']} "

    print("Installing ", cmd)
    os.system(cmd)
    sleep(60)


def env_pip_check(env_pars=None):
    from importlib import import_module

    if env_pars is None:
        env_pars = {'name': "test", 'requirement': 'requirements.txt', "import": ['tensorflow', 'sklearn']}

    flag = 0
    try:
        for f in env_pars['import']:
            import_module(f)
    except:
        flag = 1

    if flag:
        env_pip_requirement(env_pars)


def env_build(model_uri, env_pars):
    from time import sleep

    model_uri2 = model_uri.replace("/", ".")
    root = os_package_root_path()
    model_path = os.path.join(root, env_pars["model_path"])

    env_pars['name'] = model_uri2
    env_pars['python_version'] = "3.6.5"
    env_pars['file_requirement'] = model_path + "/requirements.txt"

    env_conda_build(env_pars=env_pars)
    sleep(60)

    env_pip_requirement(env_pars=env_pars)


####################################################################################################
########## TF specific #############################################################################
def tf_deprecation():
    try:
        from tensorflow.python.util import deprecation
        deprecation._PRINT_DEPRECATION_WARNINGS = False
        print("Deprecaton set to False")
    except:
        pass






####################################################################################################
###########  Utils #################################################################################
class Model_empty(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None
                 ):
        ### Model Structure        ################################
        self.model = None


def os_path_split(path) :
  return str(Path( path ).parent), str(Path( path ).name) # + str(Path( path ).suffix) 



def load(load_pars):
    p = load_pars

    if "model_keras" in p['model_uri']:
        path = os.path.abspath(p['path'] + "/../")
        name = os.path.basename(p['path']) if ".h5" in p['path'] else "model.h5"
        return load_keras(load_pars)


def save(model=None, session=None, save_pars=None):
    p = save_pars
    if "model_keras" in p['model_uri']:
        path = os.path.abspath( p['path'] + "/../")
        name = os.path.basename(p['path']) if ".h5" in p['path'] else "model.h5"
        save_keras( model, session, save_pars)





def load_tf(load_pars=""):
  """
  https://www.mlflow.org/docs/latest/python_api/mlflow.tensorflow.html#

 """
  import tensorflow as tf
  tf_graph = tf.Graph()
  tf_sess = tf.Session(graph=tf_graph)
  model_path = os.path.join(load_pars['path'], "model")
  with tf_graph.as_default():
    new_saver = tf.train.import_meta_graph(f"{model_path}.meta")
    new_saver.restore(tf_sess, tf.train.latest_checkpoint(str(Path(model_path).parent)))

  return tf_sess


def save_tf(model=None, sess=None, save_pars= None):
  import tensorflow as tf
  saver = tf.compat.v1.train.Saver()
  if not os.path.exists(save_pars['path']):
      os.makedirs(save_pars['path'], exist_ok=True)
  return saver.save(sess, os.path.join(save_pars['path'], "model"))



def load_tch(load_pars):
    import torch
    #path, filename = load_pars['path'], load_pars.get('filename', "model.pkl")
    #path = path + "/" + filename if "." not in path else path
    if os.path.isdir(load_pars['path']):
        path, filename = load_pars['path'], "model.pb"
    else:
        path, filename = os_path_split(load_pars['path'])
    model = Model_empty()
    model.model = torch.load(Path(path) / filename)
    return model


def save_tch(model=None, optimizer=None, save_pars=None):
    import torch
    if os.path.isdir(save_pars['path']):
        path, filename = save_pars['path'], "model.pb"
    else:
        path, filename = os_path_split(save_pars['path'])
    if not os.path.exists(path): os.makedirs(path, exist_ok=True)

    if save_pars.get('save_state') is not None:
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{path}/{filename}" )

    else:
        torch.save(model.model, f"{path}/{filename}")


def save_tch_checkpoint(model, optimiser, save_pars):
    import torch
    path = save_pars['checkpoint_name']
    torch.save({
        'grad_step': save_pars["grad_step"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, path )



# def load(model, optimiser, CHECKPOINT_NAME='nbeats-fiting-checkpoint.th'):
def load_tch_checkpoint(model, optimiser, load_pars):
    import torch
    CHECKPOINT_NAME = load_pars['checkpoint_name']
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
        return grad_step
    return 0



def load_pkl(load_pars):
    import cloudpickle as pickle
    return pickle.load(open( load_pars['path'], mode='rb') )


def save_pkl(model=None, session=None, save_pars=None):
    import cloudpickle as pickle
    if os.path.isdir(save_pars['path']):
        path, filename = save_pars['path'], "model.pkl"
    else:
        path, filename = os_path_split(save_pars['path'])
    if not os.path.exists(path): os.makedirs(path, exist_ok=True)
    return pickle.dump(model, open( f"{path}/{filename}" , mode='wb') )


def load_keras(load_pars, custom_pars=None):
    from tensorflow.keras.models import load_model
    if os.path.isfile(load_pars['path']):
        path, filename = os_path_split(load_pars['path']  )
    else:
        path = load_pars['path']
        filename = "model.h5"

    path_file = path + "/" + filename if ".h5" not in path else path
    model = Model_empty()
    if custom_pars:
        model.model = load_model(path_file, 
                             custom_objects={"MDN": custom_pars["MDN"],
                                             "mdn_loss_func": custom_pars["loss"]})
    else:
        model.model = load_model(path_file)
    return model


def save_keras(model=None, session=None, save_pars=None, ):
    if os.path.isdir(save_pars['path']):
        path = save_pars['path']
        filename = "model.h5"

    else:
        path, filename = os_path_split(save_pars['path'])
    if not os.path.exists(path): os.makedirs(path, exist_ok=True)
    model.model.save(str(Path(path) / filename))

def save_gluonts(model=None, session=None, save_pars=None):
    path = save_pars['path']
    if not os.path.exists(path): os.makedirs(path, exist_ok=True)
    model.model.serialize(Path(path))



def load_gluonts(load_pars=None):
    from gluonts.model.predictor import Predictor
    predictor_deserialized = Predictor.deserialize(Path( load_pars['path'] ))

    model = Model_empty()
    model.model = predictor_deserialized
    return model








"""
def path_local_setup(current_file=None, out_folder="", sublevel=0, data_path="dataset/"):
    root = os_package_root_path(__file__, sublevel=0, path_add="")

    out_path = path_norm(out_folder)
    data_path = path_norm(data_path)

    model_path = f"{out_path}/model/"
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    log(data_path, out_path, model_path)
    return data_path, out_path, model_path
"""







