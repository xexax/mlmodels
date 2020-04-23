  
# -*- coding: utf-8 -*-
"""
https://github.com/NTMC-Community/MatchZoo-py/tree/master/tutorials
https://matchzoo.readthedocs.io/en/master/model_reference.html

https://github.com/NTMC-Community/MatchZoo-py/blob/master/tutorials/classification/esim.ipynb

"""
import os, json
import importlib
import torch
import matchzoo as mz
import numpy as np
import pandas as pd
from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri, path_norm_dict

MODEL_URI = get_model_uri(__file__)

MODELS = {
    'DRMM' : importlib.import_module("matchzoo_drmm").DRMM,
    'DRMMTKS' : mz.models.DRMMTKS, 
    'ARC-I' : mz.models.ArcI,
    'ARC-II' : mz.models.ArcII, 
    'DSSM' : mz.models.DSSM, 
    'CDSSM': mz.models.CDSSM, 
    'MatchLSTM' : mz.models.MatchLSTM,
    'DUET' : mz.models.DUET,
    'KNRM' : mz.models.KNRM,
    'ConvKNRM' : mz.models.ConvKNRM, 
    'ESIM' : mz.models.ESIM,
    'BiMPM' : mz.models.BiMPM,
    'MatchPyramid' : mz.models.MatchPyramid, 
    'Match-SRNN' : mz.models.MatchSRNN,
    'aNMM' : mz.models.aNMM,
    'HBMP' : mz.models.HBMP,
    'BERT' : importlib.import_module("matchzoo_bert").Bert
}

TASKS = {
    'ranking' : mz.tasks.Ranking,
    'classification' : mz.tasks.Classification,
}

METRICS = {
    'NormalizedDiscountedCumulativeGain' : mz.metrics.NormalizedDiscountedCumulativeGain,
    'MeanAveragePrecision' : mz.metrics.MeanAveragePrecision,
    'acc' : 'acc'
}

LOSSES = {
    'RankHingeLoss' : mz.losses.RankHingeLoss,
    'RankCrossEntropyLoss' : mz.losses.RankCrossEntropyLoss
}

from pytorch_transformers import AdamW
from torch.optim import Adadelta
OPTIMIZERS = {
    'ADAMW' : lambda prm, cp : AdamW(prm, lr=cp["lr"], betas=(cp["beta1"],cp["beta2"]), eps=cp["eps"]),
    'ADADELTA' : lambda prm, cp : Adadelta(prm, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
}


def get_config_file():
    return os.path.join(os_package_root_path(__file__, 1), 'config', 'model_tch', 'Imagecnn.json')

def get_raw_dataset_wikiqa(data_pars, task):
    filter_train_pack_raw = data_pars["filter_train_data"] if "filter_train_data" in data_pars else False
    filter_dev_pack_raw = data_pars["filter_dev_data"] if "filter_dev_data" in data_pars else False
    filter_test_pack_raw = data_pars["filter_test_data"] if "filter_test_data" in data_pars else False

    train_pack_raw = mz.datasets.wiki_qa.load_data('train', task=task, filtered=filter_train_pack_raw)
    dev_pack_raw   = mz.datasets.wiki_qa.load_data('dev', task=task, filtered=filter_dev_pack_raw)
    test_pack_raw  = mz.datasets.wiki_qa.load_data('test', task=task, filtered=filter_test_pack_raw)
    return train_pack_raw, dev_pack_raw, test_pack_raw

###########################################################################################################
###########################################################################################################
class Model:
    def get_ranking_loss_from_json(self, model_pars):
        _loss = list(model_pars["loss"].keys())[0]
        _loss_params = model_pars["loss"][_loss]
        if _loss == 'RankHingeLoss':
            return LOSSES[_loss]()
        elif _loss == 'RankCrossEntropyLoss':
            return LOSSES[_loss](num_neg=_loss_params["num_neg"])

    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, out_pars=None):
        ### Model Structure        ################################

        if model_pars is None :
            self.model = None
            return self
 
        _model = model_pars['model']
        assert _model in MODELS.keys()
        _task = model_pars['task']
        assert _task in TASKS.keys()
        if _task == "ranking":
            self.task = mz.tasks.Ranking(losses=self.get_ranking_loss_from_json(model_pars))
        elif _task == "classification" :
            self.task = mz.tasks.Classification(num_classes=model_pars["num_classes"])
        else:
            raise Exception(f"Not support choice {task} yet")
        
        _metrics = model_pars['metrics']
        self.task.metrics = []
        for metric in _metrics.keys():
            metric_params = _metrics[metric]
            # Find a better way later to apply params for metric, for now hardcode.
            if metric == 'NormalizedDiscountedCumulativeGain' and metric_params != {}:
                self.task.metrics.append(METRICS[metric](k=metric_params["k"]))
            elif metric in METRICS:
                self.task.metrics.append(METRICS[metric])
            else:
                raise Exception(f"Not support choice {task_m} yet")
        
        # Get Raw Data from Dataset
        trpr, dpr, tepr = get_raw_dataset_wikiqa(data_pars, self.task)
        # Prepare model and data for preprocessing.
        self.model = MODELS[_model](model_pars, data_pars, self.task, trpr, dpr, tepr)

def get_params(param_pars=None, **kw):
    pp          = param_pars
    choice      = pp['choice']
    config_mode = pp['config_mode']
    data_path   = pp['data_path']

    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]

        ####Normalize path  : add /models/dataset/
        cf['data_pars'] = path_norm_dict(cf['data_pars'])
        cf['out_pars']  = path_norm_dict(cf['out_pars'])

        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")

def fit(model, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    model0        = model.model
    epochs        = compute_pars["epochs"]

    optimize_parameters = compute_pars["optimize_parameters"] \
        if "optimize_parameters" in compute_pars else False
    if optimize_parameters:
        model_parameters = model0.get_optimized_parameters()
    else:
        model_parameters = model0.parameters()
    
    optimizer_ = compute_pars["optimizer"]
    optimizer = OPTIMIZERS[optimizer_](model_parameters, compute_pars)

    trainer = mz.trainers.Trainer(
                device='cpu',
                model=model.model, 
                optimizer=optimizer, 
                trainloader=model0.trainloader, 
                validloader=model0.validloader, 
                epochs=epochs
            )
    trainer.run()
    return model, None

def predict(model, session=None, data_pars=None, compute_pars=None, out_pars=None):
    # get a batch of data
    model0 = model.model
    _, valid_iter = get_dataset(data_pars=data_pars)
    x_test        = next(iter(valid_iter))[0].to(device)
    ypred         = model0(x_test).detach().cpu().numpy()
    return ypred


def fit_metrics(model, data_pars=None, compute_pars=None, out_pars=None):
    pass


def save(model, session=None, save_pars=None):
    from mlmodels.util import save_tch
    save_tch(model=model.model, save_pars=save_pars)


def load(load_pars):
    from mlmodels.util import load_tch
    return load_tch(load_pars)



###########################################################################################################
###########################################################################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice":pars_choice,  "data_path":data_path,  "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
    log(  data_pars, out_pars )

    log("#### Loading dataset   #############################################")
    #xtuple = get_dataset(data_pars)


    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    model, session = fit(model, data_pars, compute_pars, out_pars)


    log("#### Predict   #####################################################")
    #ypred = predict(model, session, data_pars, compute_pars, out_pars)


    log("#### metrics   #####################################################")
    #metrics_val = fit_metrics(model, data_pars, compute_pars, out_pars)
    print(metrics_val)


    log("#### Plot   ########################################################")


    log("#### Save/Load   ###################################################")
    save_pars = { "path": out_pars["path"]  }
    save(model=model, save_pars=save_pars)
    model2 = load( save_pars )
    ypred = predict(model2, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    print(model2)



if __name__ == "__main__":
    test(data_path="model_tch/matchzoo_ranking_bert.json", pars_choice="json", config_mode="test")





