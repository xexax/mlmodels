import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from autogluon import TabularPrediction as tabular_task
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.model.predictor import Predictor

VERBOSE = False
URL_INC_TRAIN = 'https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv'
URL_INC_TEST = 'https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv'


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
def _get_dataset_from_aws(**kw):
    dt_name = kw['dt_name']
    if dt_name == 'Inc':
        if kw['train']:
            data = tabular_task.Dataset(file_path=URL_INC_TRAIN)
        else:
            data = tabular_task.Dataset(file_path=URL_INC_TEST)
        label = 'occupation'
        if kw.get('label'):
            label = kw.get('label')

        return data, label
    else:
        print(f"Not support {dt_name} yet!")


# Dataset
def get_dataset(**kw):
    if kw['dt_source'] == 'amazon_aws':
        return _get_dataset_from_aws(**kw)
    else:
        ##check whether dataset is of kind train or test
        data_path = kw['train_data_path'] if kw['train'] else kw['test_data_path']

        #### read from csv file
        if kw.get("uri_type") == "pickle":
            data_set = pd.read_pickle(data_path)
        else:
            data_set = pd.read_csv(data_path)

        ### convert to gluont format
        gluonts_ds = ListDataset(
            [{FieldName.TARGET: data_set.iloc[i].values, FieldName.START: kw['start']}
             for i in range(kw['num_series'])], freq=kw['freq'])

        if VERBOSE:
            entry = next(iter(gluonts_ds))
            train_series = to_pandas(entry)
            train_series.plot()
            save_fig = kw['save_fig']
            plt.savefig(save_fig)

        return gluonts_ds


# Model fit
def fit(model, data=None, model_pars=None, compute_pars=None, out_pars=None, session=None,
        **kwargs):
    ##loading dataset
    """
      Classe Model --> model,   model.model contains thte sub-model

    """
    if data is None or not isinstance(data, (list, tuple)):
        raise Exception("Missing data or invalid data format for fitting!")

    train_ds, label = data
    nn_options = {
        'num_epochs': compute_pars['num_epochs'],
        'learning_rate': model_pars['learning_rate'],
        'activation': model_pars['activation'],
        'layers': model_pars['layers'],
        'dropout_prob': model_pars['dropout_prob'],
    }
    gbm_options = {
        'num_boost_round': model_pars['num_boost_round'],
        'num_leaves': model_pars['num_leaves'],
    }
    predictor = model.model.fit(train_data=train_ds, label=label,
                                output_directory=out_pars['outpath'],
                                time_limits=compute_pars['time_limits'],
                                num_trials=compute_pars['num_trials'],
                                hyperparameter_tune=compute_pars['hp_tune'],
                                hyperparameters={'NN': nn_options,
                                                 'GBM': gbm_options},
                                search_strategy=compute_pars['search_strategy'])
    model.predictor = predictor
    return model


# Model p redict
def predict(model, data_pars, compute_pars=None, out_pars=None, **kwargs):
    ##  Model is class
    ## load test dataset
    data_pars['train'] = False
    test_ds, label = get_dataset(**data_pars)
    # remove label in test data if have
    if label in test_ds.columns:
        test_ds = test_ds.drop(labels=[label], axis=1)

    y_pred = model.predictor.predict(test_ds)

    ### output stats for prediction
    if VERBOSE:
        pass
    return y_pred


def metrics(model, ypred, data_pars, compute_pars=None, out_pars=None, **kwargs):
    ## load test dataset
    data_pars['train'] = False
    test_ds, label = get_dataset(**data_pars)
    y_test = test_ds[label]

    ## evaluate
    acc = model.predictor.evaluate_predictions(y_true=y_test, y_pred=ypred, auxiliary_metrics=False)
    metrics_dict = {"ACC": acc}
    return metrics_dict


###############################################################################################################
### different plots and output metric
def plot_prob_forecasts(ypred, out_pars=None):
    forecast_entry = ypred["forecasts"][0]
    ts_entry = ypred["tss"][0]

    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in
                                                      prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()


def plot_predict(item_metrics, out_pars=None):
    item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
    plt.grid(which="both")
    outpath = out_pars['outpath']
    plt.savefig(outpath)
    plt.clf()
    print('Saved image to {}.'.format(outpath))


###############################################################################################################
# save and load model helper function
class Model_empty(object):
    def __init__(self, model_pars=None, compute_pars=None):
        ## Empty model for Seaialization
        self.model = tabular_task


def save(model):
    if not model:
        print("model do not exist!")
    else:
        model.predictor.save()


def load(path):
    if not os.path.exists(path):
        print("model file do not exist!")
        return None
    else:
        model = Model_empty()
        model.predictor = tabular_task.load(path)

        #### Add back the model parameters...
        return model
