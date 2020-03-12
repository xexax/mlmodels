# -*- coding: utf-8 -*-
def predict(model, sess=None, data_pars={}, out_pars={}, compute_pars={}, **kw):
    ##### Get Data ###############################################
    reader = get_dataset(data_pars)
    train_fname = 'train.gz' if data_pars["train_type"].lower() == 'nli'\
            else 'sts-train.csv'
    examples = [ex.texts for ex in reader.get_examples(train_fname)]
    Xpred = sum(examples, [])

    #### Do prediction
    ypred = model.model.encode(Xpred)

    ### Save Results

    ### Return val
    # if compute_pars.get("return_pred_not") is not None :
    return ypred
"""

https://pypi.org/project/sentence-transformers/

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')


sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)


bert-base-nli-cls-token.zip                        14-Aug-2019 12:53           405237015
bert-base-nli-max-tokens.zip                       14-Aug-2019 12:53           405231770
bert-base-nli-mean-tokens.zip                      14-Aug-2019 09:03           405234788
bert-base-nli-stsb-mean-tokens.zip                 14-Aug-2019 12:52           405233603
bert-base-wikipedia-sections-mean-tokens.zip       14-Aug-2019 12:53           405227291
bert-large-nli-cls-token.zip                       14-Aug-2019 12:57          1243516284
bert-large-nli-max-tokens.zip                      14-Aug-2019 12:54          1243506292
bert-large-nli-mean-tokens.zip                     14-Aug-2019 12:55          1243511182
bert-large-nli-stsb-mean-tokens.zip                14-Aug-2019 12:57          1243516997
distilbert-base-nli-mean-tokens.zip                06-Dec-2019 11:09           244584363
distilbert-base-nli-stsb-mean-tokens.zip           06-Dec-2019 13:51           244582843
distiluse-base-multilingual-cased.zip              23-Jan-2020 09:51           504125638
readme.txt                                         14-Aug-2019 09:04                  82
roberta-base-nli-mean-tokens.zip                   05-Dec-2019 12:44           458913367
roberta-base-nli-stsb-mean-tokens.zip              05-Dec-2019 13:21           459100025
roberta-large-nli-mean-tokens.zip                  05-Dec-2019 12:49          1312045841
roberta-large-nli-stsb-mean-tokens.zip             05-Dec-2019 13:43          1312308420



bert-base-nli-mean-tokens: BERT-base model with mean-tokens pooling. Performance: STSbenchmark: 77.12
bert-base-nli-max-tokens: BERT-base with max-tokens pooling. Performance: STSbenchmark: 77.18
bert-base-nli-cls-token: BERT-base with cls token pooling. Performance: STSbenchmark: 76.30
bert-large-nli-mean-tokens: BERT-large with mean-tokens pooling. Performance: STSbenchmark: 79.19
bert-large-nli-max-tokens: BERT-large with max-tokens pooling. Performance: STSbenchmark: 78.32
bert-large-nli-cls-token: BERT-large with CLS token pooling. Performance: STSbenchmark: 78.29
roberta-base-nli-mean-tokens: RoBERTa-base with mean-tokens pooling. Performance: STSbenchmark: 77.42
roberta-large-nli-mean-tokens: RoBERTa-base with mean-tokens pooling. Performance: STSbenchmark: 78.58
distilbert-base-nli-mean-tokens: DistilBERT-base with mean-tokens pooling. Performance: STSbenchmark: 76.97
Trained on STS data

These models were first fine-tuned on the AllNLI datasent, then on train set of STS benchmark. They are specifically well suited for semantic textual similarity. For more details, see: sts-models.md.

bert-base-nli-stsb-mean-tokens: Performance: STSbenchmark: 85.14
bert-large-nli-stsb-mean-tokens: Performance: STSbenchmark: 85.29
roberta-base-nli-stsb-mean-tokens: Performance: STSbenchmark: 85.40
roberta-large-nli-stsb-mean-tokens: Performance: STSbenchmark: 86.31
distilbert-base-nli-stsb-mean-tokens: Performance: STSbenchmark: 84.38



Generic template for new model.
Check parameters template in models_config.json

"model_pars":   { "learning_rate": 0.001, "num_layers": 1, "size": 6, "size_layer": 128, "output_size": 6, "timestep": 4, "epoch": 2 },
"data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] },
"compute_pars": { "distributed": "mpi", "epoch": 10 },
"out_pars":     { "out_path": "dataset/", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] }






"""
import glob
import inspect
import json
import logging
import math
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import (confusion_matrix, matthews_corrcoef,
                             mean_squared_error)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm_notebook, trange




from sentence_transformers import (LoggingHandler, SentencesDataset,
                                   SentenceTransformer, losses, models,
                                   readers)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import sentence_transformers as K
from sentence_transformers import models
#####################################################################################################


VERBOSE = False



####################################################################################################
def os_module_path():
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    # sys.path.insert(0, parent_dir)
    return parent_dir


def os_file_path(data_path):
    from pathlib import Path
    data_path = os.path.join(Path(__file__).parent.parent.absolute(), data_path)
    return data_path


def os_package_root_path(filepath, sublevel=0, path_add=""):
    """
    get the module package root folder
    """
    from pathlib import Path
    path = Path(filepath).parent
    for i in range(1, sublevel + 1):
        path = path.parent

        path = os.path.join(path.absolute(), path_add)
        return path
# print("check", os_package_root_path(__file__, sublevel=1) )


def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)


class to_namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

        def get(self, key):
            return self.__dict__.get(key)


def path_setup(out_folder="", sublevel=1, data_path="dataset/"):
    data_path = os_package_root_path(__file__, sublevel=sublevel, path_add=data_path)
    out_path = os.getcwd() + "/" + out_folder
    os.makedirs(out_path, exist_ok=True)
    model_path = out_path + "/model/"
    os.makedirs(model_path, exist_ok=True)

    log(data_path, out_path, model_path)
    return data_path, out_path, model_path



####################################################################################################
class Model:
  def __init__(self, model_pars=None, data_pars=None, skip_create=False):
    ### Model Structure        ################################
    if not skip_create:
        self.EmbeddingModel = getattr(models, model_pars["embedding_model"])
        self.word_embedding_model = self.EmbeddingModel(
            model_pars["embedding_model_name"])
        self.pooling_model = models.Pooling(
            self.word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False)
        self.model = SentenceTransformer(
            modules=[self.word_embedding_model, self.pooling_model]
        )
    self.fit_metrics = {"train": {}, "test": {}}    #### metrics during training

def fit(model, data_pars={}, model_pars={}, compute_pars={},
        out_pars={}, *args, **kw):
    """

  :param model:    Class model
  :param data_pars:  dict of
  :param out_pars:
  :param compute_pars:
  :param kwargs:
  :return:
    """
    train_pars = data_pars.copy()
    train_pars.update(train=1)
    train_fname = 'train.gz' if data_pars["train_type"].lower() == 'nli'\
        else 'sts-train.csv'
    val_pars = data_pars.copy()
    val_pars.update(train=0)
    val_fname = 'dev.gz' if data_pars["test_type"].lower() == 'nli'\
        else 'sts-dev.csv'
    train_reader = get_dataset(train_pars)
    val_reader = get_dataset(val_pars)

    train_data = SentencesDataset(train_reader.get_examples(train_fname),
                                  model=model.model)
    val_data = SentencesDataset(val_reader.get_examples(val_fname),
                                model=model.model)
    train_dataloader = DataLoader(train_data, shuffle=True,
                                  batch_size=compute_pars["batch_size"])
    val_dataloader = DataLoader(val_data, shuffle=True,
                                batch_size=compute_pars["batch_size"])
    emb_dim = model.model.get_sentence_embedding_dimension()
    train_num_labels = train_reader.get_num_labels()
    train_loss = getattr(losses, compute_pars["loss"])(
        model=model.model,
        sentence_embedding_dimension=emb_dim,
        num_labels=train_num_labels
    )
    train_loss.float()
    evaluator = EmbeddingSimilarityEvaluator(val_dataloader)
    model.model.float()
    model.fit_metrics =\
        model.model.fit(train_objectives=[(train_dataloader, train_loss)],
                        evaluator=evaluator,
                        epochs=compute_pars["num_epochs"],
                        evaluation_steps=compute_pars["evaluation_steps"],
                        warmup_steps=compute_pars["warmup_steps"],
                        output_path=out_pars["model_save_path"]
                        )
    return model, None


def fit_metrics(model, **kw):
    """
       Return metrics of the model when fitted.
    """
    ddict = model.fit_metrics    
    return ddict


def predict(model, sess=None, data_pars={}, out_pars={}, compute_pars={}, **kw):
    ##### Get Data ###############################################
    reader = get_dataset(data_pars)
    train_fname = 'train.gz' if data_pars["train_type"].lower() == 'nli'\
            else 'sts-train.csv'
    examples = [ex.texts for ex in reader.get_examples(train_fname)]
    Xpred = sum(examples, [])

    #### Do prediction
    ypred = model.model.encode(Xpred)

    ### Save Results

    ### Return val
    # if compute_pars.get("return_pred_not") is not None :
    return ypred
  
def reset_model():
    pass


def save(model, out_pars):
    return torch.save(model.model, out_pars['modelpath'])

def load(out_pars={}):
    model = Model(skip_create=True)
    model.model = torch.load(out_pars['modelpath'])
    return model   



####################################################################################################
def get_dataset(data_pars=None, **kw):
    """
    JSON data_pars to get dataset
    "data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas",
    "size": [0, 0, 6], "output_size": [0, 6] },
    """
    data_path, _, _ = path_setup()
    mode = "train" if data_pars["train"] else "test"
    if data_pars[f"{mode}_type"].lower() == 'nli':
        Reader = readers.NLIDataReader
    elif data_pars[f"{mode}_type"].lower() == 'sts':
        Reader = readers.STSDataReader
    path = os.path.join(data_path, data_pars[f"{mode}_path"])
    reader = Reader(path)
    return reader

def get_params(choice="", data_path="dataset/", config_mode="test", **kw):
    if choice == "json":
        with open(data_path, encoding='utf-8') as config_f:
            config = json.load(config_f)
            c = config[config_mode]

        model_pars, data_pars  = c[ "model_pars" ], c[ "data_pars" ]
        compute_pars, out_pars = c[ "compute_pars" ], c[ "out_pars" ]
        return model_pars, data_pars, compute_pars, out_pars


    if choice == "test01":
        log("#### Path params   ##########################################")
        data_path, out_path, model_path = path_setup(out_folder="", sublevel=1,
                                                     data_path="dataset/")
        data_pars = {
            "train_path": "AllNLI",
            # one of: STS, NLI
            "train_type": "NLI",
            "test_path": "stsbenchmark",
            # one of: STS, NLI
            "test_type": "sts",
            "train": 1
        }
        model_pars = {
            "embedding_model": "BERT",
            "embedding_model_name": "bert-base-uncased",
        }
        compute_pars = {
            # "loss": "CosineSimilarityLoss",
            "loss": "SoftmaxLoss",
            "batch_size": 32,
            "num_epochs": 1,
            "evaluation_steps": 10,
            "warmup_steps": 100,
        }
        out_pars = {
            "model_save_path": "/tmp/sentence_transformers"
        }

    return model_pars, data_pars, compute_pars, out_pars



################################################################################################
def test_global(data_path="dataset/", model_uri="model_tch/transformer_sentence.py", pars_choice="test01", reset=True):
    ###loading the command line arguments
    #model_uri = "model_xxxx/yyyy.py"

    log("#### Module init   ############################################")
    from mlmodels.models import module_load
    module = module_load(model_uri)
    log(module)


    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = module.get_params(choice=pars_choice,
                                                                      data_path=data_path)


    log("#### Model init   ############################################")
    from mlmodels.models import model_create 
    model = model_create(module, model_pars)
    log(model)


    log("#### Fit   ########################################################")
    model, session = module.fit(model, data_pars, model_pars, compute_pars, out_pars)


    log("#### Predict   ####################################################")
    ypred = module.predict(model, session, data_pars, compute_pars, out_pars)
    print(ypred)


    log("#### Get  metrics   ################################################")
    metrics_val = module.fit_metrics(model, data_pars, compute_pars, out_pars)


    log("#### Save/Load   ###################################################")
    module.save(model, out_pars['modelpath'])
    model2 = module.load(out_pars['modelpath'])
    #     ypred = predict(model2, data_pars, compute_pars, out_pars)
    #     metrics_val = metrics(model2, ypred, data_pars, compute_pars, out_pars)
    print(model2)




def test_global2(data_path="dataset/", model_uri="model_tch", pars_choice="json", reset=True):
    ###loading the command line arguments
    model_uri = "model_xxxx/yyyy.py"


    log("#### Module init   #################################################")
    from mlmodels.models import module_load
    module = module_load(model_uri)
    log(module)


    log("#### Loading params   ##############################################")
    from mlmodels.models import get_params as get_params_batch
    model_pars, data_pars, compute_pars, out_pars = get_params_batch(module, choice=pars_choice,
                                                                    data_path=data_path)

    log("#### Model init, fit   ############################################")
    from mlmodels.models import module_load_full 
    module, model = module_load_full(model_uri, model_pars)
    log(module, model)


    log("#### Fit   ########################################################")
    from mlmodels.models import fit as fit_batch
    model, session = fit_batch(model, module, compute_pars, data_pars, out_pars)


    log("#### Predict   ####################################################")
    ypred = predict_batch(model, module, session, compute_pars, data_pars, out_pars)
    print(ypred)


    log("#### Mtrics   ################################################")
    from mlmodels.models import  metrics as metrics_batch 
    metrics_val = metrics_batch(model, module, session, compute_pars, data_pars,  out_pars)


    log("#### Save/Load   ###################################################")
    from mlmodels.models import load_batch, save_batch
    save_batch(model, module, session, out_pars['modelpath'])
    model2 = load_batch(out_pars['modelpath'])
    #     ypred = predict(model2, data_pars, compute_pars, out_pars)
    #     metrics_val = metrics(model2, ypred, data_pars, compute_pars, out_pars)
    print(model2)






def test(data_path="dataset/", pars_choice="test01"):
    ### Local test

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=pars_choice,
                                                               data_path=data_path)

    log("#### Loading dataset   #############################################")
    Xtuple = get_dataset(data_pars)


    log("#### Model init, fit   #############################################")
    model = Model(model_pars, compute_pars)
    model = fit(model, data_pars, model_pars, compute_pars, out_pars)


    log("#### save the trained model  #######################################")
    save(model, out_pars["modelpath"])


    log("#### Predict   #####################################################")
    ypred = predict(model, data_pars, compute_pars, out_pars)


    log("#### metrics   #####################################################")
    metrics_val = fit_metrics(model, ypred, data_pars, compute_pars, out_pars)
    print(metrics_val)


    log("#### Save/Load   ###################################################")
    save(model, out_pars['modelpath'])
    model2 = load(out_pars['modelpath'])
    #     ypred = predict(model2, data_pars, compute_pars, out_pars)
    #     metrics_val = metrics(model2, ypred, data_pars, compute_pars, out_pars)
    print(model2)


if __name__ == '__main__':
    VERBOSE = True
    
    ### Local
    test(pars_choice="json")
    test(pars_choice="test01")

    ### Global mlmodels
    test_global(pars_choice="json", out_path= test_path,  reset=True)
