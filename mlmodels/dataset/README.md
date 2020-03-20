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



Wednesday April :

   From  9:00 AM (9:00) Tokyo Time =   until anytime 
      (5:00 PM (17:00) Previous Day San Francisco Time)

   From  10 AM (Tokyo tme),  (6pm DF time).


   From 1:00 AM (1:00) Tokyo Time (  9:00 AM (9:00) Previous Day San Francisco Time )


Thursday April 2:
   From  9:00 AM (9:00) Tokyo Time =   until anytime 
      (5:00 PM (17:00) Previous Day San Francisco Time)

  From  10 AM (Tokyo tme),  (6pm DF time).










    to login with a token:
        http://localhost:8889/?token=


ded64cfa8d4cf6d08ad190a74431341157eac09e73d36fda&token=ded64cfa8d4cf6d08ad190a74431341157eac09e73d36fda



```

        data_pars ={
            "path"            : 
            "path_type"   :  "local/absolute/web"


            "data_type"   :   "text" / "recommender"  / "timeseries" /"image"
            "data_split"  : {"istrain" :  1   , "split_size" : 0.5, "randomseed" : 1   },


            "data_loader"      :   "mlmodels.data:pd_reader",
            "data_loader_pars" :   { "sequential" : 1  },


            "data_preprocessor" : "mlmodels.model_keras.prepocess:process",
            "data_preprocessor_pars" :  {  },


            "size" : [0,1,2],
            "output_size": [0, 6]            
          }




```




