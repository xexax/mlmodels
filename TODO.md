## TODO

###  Write use cases of mlmodels
    USECASE.md

### Renormalize signature

     fit(model, data_pars, model_pars, compute_pars, )         : model, session
    
     predict(model, sess, data_pars, compute_pars, out_pars )  : ypred : numpy
    
     metric(ytrue, ypred, yproba, model=None data_pars, compute_pars, out_pars )  : ddict 
    
     get_params(choice:str, dataset:str, mode:str)   : 
    
     get_dataset(data_pars:str)  : 
    
     test(choice:str)    :      
    
     test2(choice:str)   :   
    
     save(model, path:str)    : save the model
     load(path:str)           : load the trained model


###  data pre-process and data visualization utils such as 
    https://course.fast.ai/datasets





