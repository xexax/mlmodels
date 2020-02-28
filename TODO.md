


### Renormalize signature





 fit(model, data_pars, model_pars, compute_pars, )         : model, session
 predict(model, sess, data_pars, compute_pars, out_pars )  : ypred : numpy
 metric(ytrue, ypred, yproba, data_pars, compute_pars, out_pars )  : ddict 

 get_params(choice="", dataset="", mode)   : 
 get_dataset(data_pars)                                    : load dataset
 test()                                                    : example running the model     
 test2()                                                   : example running the model in global settings  

 save(model, path)                                         : save the model
 load(path)                                                : load the trained model
