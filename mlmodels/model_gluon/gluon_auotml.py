# -*- coding: utf-8 -*-
"""
AutGluon

https://autogluon.mxnet.io/tutorials/tabular_prediction/tabular-quickstart.html



"""

import autogluon as ag

from mlmodels.model_gluon.util_autogluon import *

"""

# First install package from terminal:  pip install mxnet autogluon

from autogluon import TabularPrediction as task
train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
predictor = task.fit(train_data=train_data, label='class')
performance = predictor.evaluate(test_data)






import autogluon as ag
from autogluon import TabularPrediction as task



train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
train_data = train_data.head(500) # subsample 500 data points for faster demo
print(train_data.head())


label_column = 'class'
print("Summary of class variable: \n", train_data[label_column].describe())



dir = 'agModels-predictClass' # specifies folder where to store trained models
predictor = task.fit(train_data=train_data, label=label_column, output_directory=dir)


test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
y_test = test_data[label_column]  # values to predict
test_data_nolab = test_data.drop(labels=[label_column],axis=1) # delete label column to prove we're not cheating
print(test_data_nolab.head())


from autogluon import TabularPrediction as task
predictor = task.fit(train_data=task.Dataset(file_path=<file-name>), label_column=<variable-name>)

results = predictor.fit_summary()

print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon categorized the features as: ", predictor.feature_types)











"""


########################################################################################################################
#### Model defintion
class Model(object):
    def __init__(self, model_pars=None, compute_pars=None):
        ## Empty model for Seaialization
        if model_pars is None and compute_pars is None:
            self.model = None

        else:
            if model_pars['model_type'] == 'tabular':
                self.model = tabular_task


########################################################################################################################
def get_params(choice=0, data_path="dataset/", **kw):
    if choice == 0:
        log("#### Path params   ################################################")
        data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
        out_path = os.getcwd() + "/GLUON_gluon/"
        os.makedirs(out_path, exist_ok=True)
        model_path = os.getcwd() + "/GLUON/model_gluon/"
        os.makedirs(model_path, exist_ok=True)
        log(data_path, out_path, model_path)

        data_pars = {"train": True, "dt_source": "amazon_aws", "dt_name": "Inc"}

        log("#### Model params   ################################################")
        model_pars = {"model_type": "tabular",
                      'learning_rate': ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),
                      'activation': ag.space.Categorical('relu', 'softrelu', 'tanh'),
                      # activation function used in NN (categorical hyperparameter, default = first entry)
                      'layers': ag.space.Categorical([100], [1000], [200, 100],
                                                     [300, 200, 100]),
                      # Each choice for categorical hyperparameter 'layers' corresponds to list of sizes for each NN layer to use
                      'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),
                      # dropout probability (real-valued hyperparameter)
                      # specifies non-default hyperparameter values for neural network models
                      'num_boost_round': 100,
                      # number of boosting rounds (controls training time of GBM models)
                      'num_leaves': ag.space.Int(lower=26, upper=66, default=36)
                      }

        compute_pars = {"hp_tune": True,
                        'num_epochs': 10,
                        # number of leaves in trees (integer hyperparameter)
                        "time_limits": 2 * 60,  # train various models for ~2 min
                        "num_trials": 5,
                        # try at most 3 different hyperparameter configurations for each type of model
                        "search_strategy": 'skopt',
                        # to tune hyperparameters using SKopt Bayesian optimization routine
                        }

        out_pars = {"outpath": out_path}

    return model_pars, data_pars, compute_pars, out_pars


########################################################################################################################
def test(data_path="dataset/", pars_choice=0):
    ### Local test

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=pars_choice,
                                                               data_path=data_path)

    log("#### Loading dataset   #############################################")
    gluon_ds = get_dataset(**data_pars)

    log("#### Model init, fit   #############################################")
    model = Model(model_pars, compute_pars)
    # model=m.model    ### WE WORK WITH THE CLASS (not the attribute GLUON )
    model = fit(model, gluon_ds, model_pars, compute_pars, out_pars)

    log("#### save the trained model  #############################################")
    # save(model, data_pars["modelpath"])

    log("#### Predict   ####################################################")
    ypred = predict(model, data_pars, compute_pars, out_pars)

    log("#### metrics   ####################################################")
    metrics_val = metrics(model, ypred, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   #######################################################")

    log("#### Save/Load   ##################################################")
    save(model)
    model2 = load(out_pars['outpath'])
    #     ypred = predict(model2, data_pars, compute_pars, out_pars)
    #     metrics_val = metrics(model2, ypred, data_pars, compute_pars, out_pars)
    print(model2)


if __name__ == '__main__':
    VERBOSE = True
    test(pars_choice=0)
