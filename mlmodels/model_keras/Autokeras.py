
# -*- coding: utf-8 -*-
"""
https://autokeras.com/examples/imdb/
"""

import os
import json
from copy import deepcopy
import numpy as np
import pandas as pd
import autokeras as ak

from keras.models import load_model

from keras.datasets import imdb, mnist


############################################################################################################
from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri, path_norm_dict

MODEL_URI = get_model_uri(__file__)


def get_config_file():
    return os.path.join(os_package_root_path(__file__, 1), 'config', 'model_keras', 'Imagecnn.json')


MODELS = {
    'text_classifier'      : ak.TextClassifier,
    'image_clasifier'      : ak.ImageClassifier,
    "tabular_classifier"   : ak.StructuredDataClassifier,
     "tabular_regressor":    ak.StructuredDataRegressor

}




###########################################################################################################
###########################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, out_pars=None):
        self.model_pars   = deepcopy(model_pars)
        self.data_pars    = deepcopy(data_pars)
        self.compute_pars = deepcopy(compute_pars)


        if model_pars is None:
            self.model = None
            return None


        if model_pars["model_name"]  == "tabular_regressor" :
            self.model = ak.StructuredDataRegressor(** model_pars["model_pars"],column_types=data_pars["data_type"])

        else :    
           self.model  = MODELS[  model_pars['model_name'] ](  ** model_pars["model_pars"]  )

       
        """  
        # initalize model according to the type
        if model_pars["model_name"] == "text":
            # Initialize the TextClassifier
            self.model = ak.TextClassifier( ** model_pars["model_pars"])
        elif model_pars["model_name"] == "vision":
            # Initialize the ImageClassifier.
            self.model = ak.ImageClassifier(** model_pars["model_pars"])
        elif model_pars["model_name"] == "tabular_classifier":
            # Initialize the classifier.
            self.model = ak.StructuredDataClassifier(** model_pars["model_pars"])


        elif model_pars["model_name"] == "tabular_regressor":
            self.model = ak.StructuredDataRegressor(** model_pars["model_pars"],column_types=data_pars["data_type"])
        """  


def get_params(param_pars=None, **kw):
    pp = param_pars
    choice = pp['choice']
    config_mode = pp['config_mode']
    data_path = pp['data_path']

    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]

        ####Normalize path  : add /models/dataset/
        cf['data_pars'] = path_norm_dict(cf['data_pars'])
        cf['out_pars'] = path_norm_dict(cf['out_pars'])

        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")



def get_dataset_imbd(data_pars):

    # Load the integer sequence the IMDB dataset with Keras.
    index_offset = 3  # word index offset
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=data_pars["num_words"],
                                                          index_from=index_offset)
    y_train               = y_train.reshape(-1, 1)
    y_test                = y_test.reshape(-1, 1)
    # Prepare the dictionary of index to word.
    word_to_id            = imdb.get_word_index()
    word_to_id            = {k: (v + index_offset) for k, v in word_to_id.items()}
    word_to_id["<PAD>"]   = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"]   = 2
    id_to_word = {value: key for key, value in word_to_id.items()}
    # Convert the word indices to words.
    x_train = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_train))
    x_test = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_test))
    x_train = np.array(x_train, dtype=np.str)
    x_test = np.array(x_test, dtype=np.str)
    return x_train, y_train, x_test, y_test


def get_dataset_titanic(data_pars): 
    # Preparing training data.
    train_data_path = data_pars['train_data_path']
    test_data_path  = data_pars['test_data_path']
    train_data_path = path_norm( train_data_path )
    test_data_path  = path_norm( test_data_path )
    x_train         = pd.read_csv(train_data_path)
    y_train         = x_train.pop('survived')
    # Preparing testing data.
    x_test          = pd.read_csv(test_data_path)
    y_test          = x_test.pop('survived')
    return x_train, y_train, x_test, y_test


def get_dataset_auto_mpg(data_pars):
    column_names = data_pars["column_names"]
    dataset_path = data_pars["dataset_path"]
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                        na_values = "?", comment='\t',
                        sep=" ", skipinitialspace=True)
    target_col    = data_pars["target_col"]
    dataset       = raw_dataset.copy()
    dataset       = dataset.dropna()
    data_type     = (len(column_names )-1) * ['numerical'] + ['categorical']
    data_type     = dict(zip(column_names , data_type))
    train_dataset = dataset.sample(frac=1-data_pars["validation_split"],random_state=42)
    test_dataset  = dataset.drop(train_dataset.index)
    return train_dataset.drop(columns=[target_col]), train_dataset[target_col], test_dataset.drop(columns=[target_col]),test_dataset[target_col]



def get_dataset(data_pars=None):

    if data_pars['dataset'] == 'IMDB':
        x_train, y_train, x_test, y_test = get_dataset_imbd(data_pars)
        
    elif data_pars['dataset'] == "MNIST":
        (x_train, y_train), (x_test, y_test)  = mnist.load_data()
        
    elif data_pars['dataset'] == "Titanic Survival Prediction":
        x_train, y_train, x_test, y_test = get_dataset_titanic(data_pars)
    elif data_pars["dataset"] == "Auto MPG Data Set":
        x_train, y_train, x_test, y_test = get_dataset_auto_mpg(data_pars)
        
    else:
        raise Exception("Dataloader not implemented")
        exit
    return x_train, y_train, x_test, y_test


def fit(model, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    x_train, y_train, _, _ = get_dataset(data_pars)
    print(type(x_train))
    print(type(y_train))
    os.makedirs(out_pars["checkpointdir"], exist_ok=True)
    model.fit(x_train,
              y_train,
              # Split the training data and use aportion as validation data.
              validation_split=data_pars["validation_split"],
              epochs=compute_pars['epochs']
              )

    return model


def predict(model, session=None, data_pars=None, compute_pars=None, out_pars=None):
    # get a batch of data
    _, _, x_test, y_test = get_dataset(data_pars)
    predicted_y = model.predict(x_test)
    return predicted_y


def fit_metrics(model, data_pars=None, compute_pars=None, out_pars=None):
    _, _, x_test, y_test = get_dataset(data_pars)
    return model.evaluate(x_test, y_test)


def save(model, session=None, save_pars=None,config_mode="test"):
    model.save(os.path.join(save_pars["checkpointdir"],f'{config_mode}.h5'))


def load(load_pars,config_mode="test"):
    return load_model(os.path.join(load_pars["checkpointdir"],f'{config_mode}.h5'), custom_objects=ak.CUSTOM_OBJECTS)


###########################################################################################################
###########################################################################################################
def test_single(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice": pars_choice, "data_path": data_path, "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
    log(data_pars, out_pars)

    log("#### Loading dataset   #############################################")
    #xtuple = get_dataset(data_pars)

    log("#### Model init, fit   #############################################")
    model = Model(model_pars, data_pars, compute_pars)
    fitted_model = fit(model.model, data_pars, compute_pars, out_pars)

    log("#### Predict   #####################################################")
    ypred = predict(fitted_model, data_pars, compute_pars, out_pars)
    print(ypred[:10])

    log("#### metrics   #####################################################")
    metrics_val = fit_metrics(fitted_model, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   ########################################################")

    log("#### Save/Load   ###################################################")
    ## Export as a Keras Model.
    save_model = fitted_model.export_model()
    save(model=save_model, save_pars=out_pars, config_mode=config_mode)
    loaded_model = load( out_pars, config_mode)
    ypred = predict(loaded_model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    print(ypred[:10])



    

def test() :
    ll = ["tabular_regressor","vision" , "text" , "tabular_classifier" ]

    for t in ll  :
        test_single(data_path="model_keras/Autokeras.json", pars_choice="json", config_mode=t)



if __name__ == '__main__':
    VERBOSE = True

    test()
