import autogluon as ag

config = {
    "test": {
        "data_pars": {"train": True, "dt_source": "amazon_aws", "dt_name": "Inc"},

        "model_pars": {"model_type": "tabular",
                      "learning_rate": ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),
                      "activation": ag.space.Categorical('relu', 'softrelu', 'tanh'),
                      "layers": ag.space.Categorical([100], [1000], [200, 100],
                                                     [300, 200, 100]),
                      'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),
                      'num_boost_round': 100,
                      'num_leaves': ag.space.Int(lower=26, upper=66, default=36)
                      },

        "compute_pars": {"hp_tune": True,
                        'num_epochs': 10,
                        "time_limits": 2 * 60,
                        "num_trials": 5,
                        "search_strategy": 'skopt',
                        }

    },
    "prod": {
        "data_pars": {"train": True, "dt_source": "amazon_aws", "dt_name": "Inc"},

        "model_pars": {"model_type": "tabular",
                      "learning_rate": ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),
                      "activation": ag.space.Categorical('relu', 'softrelu', 'tanh'),
                      "layers": ag.space.Categorical([100], [1000], [200, 100],
                                                     [300, 200, 100]),
                      'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),
                      'num_boost_round': 100,
                      'num_leaves': ag.space.Int(lower=26, upper=66, default=36)
                      },

        "compute_pars": {"hp_tune": True,
                        'num_epochs': 10,
                        "time_limits": 2 * 60,
                        "num_trials": 5,
                        "search_strategy": 'skopt',
                        }
    }
}