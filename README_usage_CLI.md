# CLI tools: package provide below tools 
```bash
- ml_models
- ml_optim    
- ml_test
- init
```

### Command Line Operations
```bash
ml_models 
    model_list      : list all supported models
    testall         : test all modules inside model_tf
    test            :  test a certain module inside model_tf
    fit             :  wrap fit generic method
    predict         :  predict  using a pre-trained model and some data
    generate_config :  generate config file from code source

  ## fit, predict options  
  --model_uri     model_tf.1_lstm
  --save_folder   myfolder/
  --config_file   myfile.json
  --config_mode   "test"
```
#### ml_models sample operations:
```bash
ml_models generate_config --model_uri model_tf.1_lstm.py  --save_folder "c:\myconfig"

ml_models test --model_uri model_tf/1_lstm.py
```

```bash
ml_optim
    test      :  Test the hyperparameter optimization for a specific model
    search    :  search for the best hyperparameters of a specific model
    Options: 
    --config_file   myfile.json
    --config_mode   "test"

ml_test
    search    :  search for the best hyperparameters of a specific model

#### Model param search test
    ml_optim --do test


#### For normal optimization search method
    ml_optim search --ntrials 1  --config_file optim_config.json --optim_method normal
    ml_optim search --ntrials 1  --config_file optim_config.json --optim_method prune  ###### for pruning method

    ml_optim test --modelname model_tf.1_lstm.py
    ml_optim search --modelname model_tf.1_lstm.py
```