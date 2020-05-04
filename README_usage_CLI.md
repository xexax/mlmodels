## Command Line Tools
```bash
- ml_models    :  Running model fit/predict
- ml_optim     :  Hyper-parameter search
- ml_benchmark :  Benchmark

- ml_test      :  Testing for developpers.


```


## How to use CLI


### ml_models
```bash
ml_models   
    model_list  :  list all models in the repo                            
    testall     :  test all modules inside model_tf
    test        :  test a certain module inside model_tf
    fit         :  wrap fit generic m    ethod
    predict     :  predict  using a pre-trained model and some data
    generate_config  :  generate config file from code source
    

  ##  fit  
     --model_uri     model_tf.1_lstm
     --save_folder   myfolder/
     --config_file   myfile.json
     --config_mode   "test"


  ## predict  
     --model_uri     model_tf.1_lstm
     --save_folder   myfolder/
     --config_file   myfile.json
     --config_mode   "test"


#### generate config file
    ml_models  generate_config  --model_uri model_tf.1_lstm.py  --save_folder "c:\myconfig"

#### TensorFlow LSTM model
    ml_models  test  --model_uri model_tf/1_lstm.py  


#### Custom  Models by URI
    ml_models test  --model_uri "D:\_devs\Python01\gitdev\mlmodels\mlmodels\model_tf.1_lstm.py"


```




### ml_optim
```bash

ml_optim 
    test      :  Test the hyperparameter optimization for a specific model
    test_all  :  TODO, Test all
    search    :  search for the best hyperparameters of a specific model



#### For normal optimization search method
    ml_optim search --ntrials 1  --config_file optim_config.json --optim_method normal


###### for pruning method
    ml_optim search --ntrials 1  --config_file optim_config.json --optim_method prune  


###### Using Model default params
    ml_optim search --modelname model_tf.1_lstm.py  


###### Using Model default params
    ml_optim test  --modelname model_tf.1_lstm.py  


```



### ml_benchmark
```bash
## Benchmark model

#### One Single file for all models
ml_benchmark  dataset/json/benchmark.json  --path_json  dataset/json/benchmark_timeseries/test02/model_list.json
     

#### Many json                            
ml_benchmark  dataset/json/benchmark.json  --path_json  dataset/json/benchmark_timeseries/test01/

    


```






### ml_distributed : Distributed training on Pytorch Horovod
```bash
### Work in Progress

#### Distributed Pytorch on CPU (using Horovod and MPI on Linux, 4 processes)  in model_tch/mlp.py
    mlmodels/distri_torch_mpirun.sh   4    model_tch.mlp    mymodel.json

    ml_distributed  fit   --n_node 4    --model_uri model_tch.mlp    --model_json mymodel.json




```



### ml_test
```bash
### For Developpers





```




## Example in Colab :

https://colab.research.google.com/drive/1u6ZUrBExDY9Jr6HA7kKutVKoP5RQfvRi#scrollTo=4qtLQiaCaDaU






