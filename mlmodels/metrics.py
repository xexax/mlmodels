






def sk_metrics_eval(ytrue, ypred, ypred_proba, metrics=["f1_macro", "accuracy", "precision_macro", "recall_macro"] ) :

  for metric in  metrics :
    metric_val = score( ytest, scoring= metric, cv=3)


    for i, metric_val_i in enumerate(metric_val):
       entries.append((model_name, i, metric, metric_val_i ))
  cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', "metric", 'metric_val'])
  return cv_df




