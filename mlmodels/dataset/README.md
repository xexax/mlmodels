#### Datasets



```
dataset/timeseries/
   Time series related



dataset/text/
   Time series related


dataset/tabular/
   Tabular related


dataset/recommender/
   Recommender


```







```

        data_pars ={
            "path"            : 
            "path_type"   :  "local/absolute/web"

            "data_type"   :   "text" / "recommender"  / "timeseries" /"image"
            "data_split"  : {"istrain" :  1   , "split_size" : 0.5, "randomseed" : 1   },

            "data_loader"      :   "mlmodels.data.pd_reader",
            "data_loader_pars" :   { "ok"  },

            "data_preprocessor" : "mlmodels.model_keras.prepocess:process",
            "data_preprocessor_pars" :  {  },

            "size" : [0,1,2],
            "output_size": [0, 6]            
          }




```




