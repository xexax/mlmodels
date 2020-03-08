## TODO



### List of potential models

https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch/tree/master/Model_CharCNN


https://github.com/chaitjo/character-level-cnn


https://gluon-nlp.mxnet.io/model_zoo/word_embeddings/index.html


http://nlp_architect.nervanasys.com/absa_solution.html


https://github.com/golsun/SpaceFusion









###  Add use cases of mlmodels
    USECASE.md



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


###  data pre-process and data visualization utils such as 
    https://course.fast.ai/datasets





