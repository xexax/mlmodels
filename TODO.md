## TODO


TODO
```

Add config json in .mlmodels 

session_path : store the session 
model_artifact : pre-trained model 
model_code : model code 
log_path : path of the logs 
working_path : path where logs, ... stored.


ml_models --do setup --json_file myconfig.json

Create dataset readme.md timseries text tabular recommender

get_dataset(data_uri="", data_type= "pandas" )






List of potential models
https://github.com/asyml/texar/tree/master/examples

https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/

https://github.com/graknlabs/kglib#knowledge-graph-tasks

https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch/tree/master/Model_CharCNN

https://github.com/chaitjo/character-level-cnn

https://github.com/benedekrozemberczki/awesome-graph-classification#deep-learning

https://gluon-nlp.mxnet.io/model_zoo/word_embeddings/index.html

http://nlp_architect.nervanasys.com/absa_solution.html

https://github.com/golsun/SpaceFusion

```

### Renormalize signature

     fit(model, data_pars, compute_pars, out_pars )         : model, session
    
     predict(model, sess, data_pars, compute_pars, out_pars )  : ypred : numpy
    
     metric(ytrue, ypred, yproba, model=None, data_pars=None, compute_pars=None, out_pars=None )  : ddict 
    
     get_params(choice:str, dataset:str, mode:str)   : 
    
     get_dataset(data_pars:str)  : 
    
     test(choice:str)    :      
    
     test2(choice:str)   :   
    
     save(model, path:str)    : save the model
     load(path:str)           : load the trained model


#### Model to Add:





###  data pre-process and data visualization utils such as 
    https://course.fast.ai/datasets





