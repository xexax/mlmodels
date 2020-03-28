
    <html>
    <body>

    [mlmodels\data.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\data.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\data.py)
[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\data.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\data.py
)

[download_data(data_pars,   )](https://github.com/arita37/mlmodels/tree/dev/mlmodels\data.py)
[get_dataset(data_pars,   )](https://github.com/arita37/mlmodels/tree/dev/mlmodels\data.py)

[import_data(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\data.py
)

[import_data_dask(  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\data.py
)

[import_data_fromfile(  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\data.py
)

[import_data_tch( name="", mode="train", node_id=0, data_folder_root="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\data.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\data.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\data.py
)

[mlmodels\distri_torch.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\distri_torch.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\distri_torch.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\distri_torch.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\distri_torch.py
)

[load_arguments(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\distri_torch.py
)

[metric_average(val, name,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\distri_torch.py
)

[test(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\distri_torch.py
)

[train(epoch,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\distri_torch.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\distri_torch.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\distri_torch.py
)

[mlmodels\models.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[cli_load_arguments( config_file=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[config_generate_json(modelname,  to_path="ztest/new_model/",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[config_get_pars(config_file,  config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[config_init( to_path=".",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[config_model_list( folder=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[fit(module, model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwarg)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[fit_metrics(module, model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwarg)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[get_params(module, params_pars,   **kwarg)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[load(module, load_pars,   **kwarg)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[main(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[metrics(module, model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwarg)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[model_create(module,  model_pars=None, data_pars=None, compute_pars=None,  **kwarg)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[module_env_build( model_uri="", verbose=0, env_build=0,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[module_load( model_uri="", verbose=0, env_build=0,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[module_load_full( model_uri="", model_pars=None, data_pars=None, compute_pars=None, choice=None,  **kwarg)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[os_folder_copy(src, dst,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[predict(module, model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwarg)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[save(module, model, session, save_pars,   **kwarg)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[test(modelname,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[test_all( folder=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[test_api( model_uri="model_xxxx/yyyy.py", param_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[test_global(modelname,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[test_module( model_uri="model_xxxx/yyyy.py", param_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\models.py
)

[mlmodels\optim.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\optim.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\optim.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\optim.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\optim.py
)

[cli_load_arguments( config_file=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\optim.py
)

[main(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\optim.py
)

[optim( model_uri="model_tf.1_lstm.py", hypermodel_pars={}, model_pars={}, data_pars={}, compute_pars={}, out_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\optim.py
)

[optim_optuna( model_uri="model_tf.1_lstm.py", hypermodel_pars={"engine" :{},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\optim.py
)

[post_process_best(model, model_uri, model_pars_update, data_pars, compute_pars, out_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\optim.py
)

[test_all(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\optim.py
)

[test_fast( ntrials=2,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\optim.py
)

[test_json( path_json="", config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\optim.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\optim.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\optim.py
)

[mlmodels\pipeline.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[Pipe.__init__(self, pipe_list, in_pars, out_pars,  compute_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[Pipe.get_checkpoint(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[Pipe.get_fitted_pipe_list(self,  key="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[Pipe.get_model_path(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[Pipe.run(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[drop_cols(df,  cols=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[generate_data(df,  num_data=0, means=[], cov=[[1, 0],  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[get_params( choice="", data_path="dataset/", config_mode="test",  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[load_model(path,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[log( n=0, m=1,  *s)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[os_package_root_path(filepath,  sublevel=0, path_add="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[pd_concat(df1, df2, colid1,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[pd_na_values(df,  cols=None, default=0.0,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[pipe_checkpoint(df,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[pipe_load(df,   **in_pars)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[pipe_merge(in_pars, out_pars,  compute_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[pipe_run_inference(pipe_list, in_pars, out_pars,  compute_pars=None, checkpoint=True,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[pipe_split(in_pars, out_pars, compute_pars,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[save_model(model, path,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[test( data_path="/dataset/", pars_choice="colnum",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\pipeline.py
)

[mlmodels\util.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[Model_empty.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[to_namespace.__init__(self, adict,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[to_namespace.get(self, key,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[config_load_root(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[config_path_dataset(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[config_path_pretrained(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[config_set(ddict2,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[env_build(model_uri, env_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[env_conda_build( env_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[env_pip_check( env_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[env_pip_requirement( env_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[get_model_uri(file,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[get_recursive_files(folderPath,  ext='/*model*/*.py',  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[get_recursive_files2(folderPath, ext,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[get_recursive_files3(folderPath, ext,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[load(load_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[load_config(args, config_file, config_mode,  verbose=0,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[load_gluonts( load_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[load_keras(load_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[load_pkl(load_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[load_tch(load_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[load_tch_checkpoint(model, optimiser, load_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[load_tf( load_pars="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[log( n=0, m=1,  *s)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[model_get_list( folder=None, block_list=[],  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[os_file_current_path(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[os_folder_copy(src, dst,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[os_get_file( folder=None, block_list=[], pattern=r'*.py',  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[os_package_root_path( filepath="", sublevel=0, path_add="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[os_path_split(path,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[params_json_load(path,  config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[path_norm( path="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[path_norm_dict(ddict,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[save( model=None, session=None, save_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[save_gluonts( model=None, session=None, save_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[save_keras( model=None, session=None, save_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[save_pkl( model=None, save_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[save_tch( model=None, optimizer=None, save_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[save_tch_checkpoint(model, optimiser, save_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[save_tf( model=None, sess=None, save_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[test_module( model_uri="model_tf/1_lstm.py", data_path="dataset/", pars_choice="json", reset=True,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[tf_deprecation(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[val(x, xdefault,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\util.py
)

[mlmodels\ztest.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest.py
)

[cli_load_arguments( config_file=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest.py
)

[main(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest.py
)

[os_file_current_path(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest.py
)

[test_all(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest.py
)

[test_custom(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest.py
)

[test_import(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest.py
)

[test_list(mlist,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest.py
)

[test_model_structure(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest.py
)

[mlmodels\ztest_structure.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest_structure.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest_structure.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest_structure.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest_structure.py
)

[code_check( sign_list=None, model_list=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest_structure.py
)

[find_in_list(x, llist,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest_structure.py
)

[get_recursive_files(folderPath,  ext='/*model*/*.py',  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest_structure.py
)

[log( n=0, m=1,  *s)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest_structure.py
)

[main(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest_structure.py
)

[model_get_list( folder=None, block_list=[],  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest_structure.py
)

[os_file_current_path(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest_structure.py
)

[os_package_root_path(filepath,  sublevel=0, path_add="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest_structure.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest_structure.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\ztest_structure.py
)

[mlmodels\__init__.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\__init__.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\__init__.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\__init__.py
)

[mlmodels\model_chatbot\__init__.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\__init__.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\__init__.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\__init__.py
)

[mlmodels\model_chatbot\diag_gpt\Chatbot_run.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\Chatbot_run.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\Chatbot_run.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\Chatbot_run.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\Chatbot_run.py
)

[generate(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\Chatbot_run.py
)

[get_bot_response(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\Chatbot_run.py
)

[home(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\Chatbot_run.py
)

[recalc( p=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\Chatbot_run.py
)

[reinput(user_msg,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\Chatbot_run.py
)

[top_p_filtering(logits,  top_p=0.9, filter_value=-float('Inf'),  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\Chatbot_run.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\Chatbot_run.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\Chatbot_run.py
)

[mlmodels\model_chatbot\diag_gpt\myChatbot.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\myChatbot.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\myChatbot.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\myChatbot.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\myChatbot.py
)

[generate(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\myChatbot.py
)

[get_bot_response(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\myChatbot.py
)

[home(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\myChatbot.py
)

[recalc( p=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\myChatbot.py
)

[reinput(user_msg,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\myChatbot.py
)

[top_p_filtering(logits,  top_p=0.9, filter_value=-float('Inf'),  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\myChatbot.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\myChatbot.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_chatbot\diag_gpt\myChatbot.py
)

[mlmodels\model_dev\__init__.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\__init__.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\__init__.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\__init__.py
)

[mlmodels\model_dev\dev\ml_mosaic.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\dev\ml_mosaic.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\dev\ml_mosaic.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\dev\ml_mosaic.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\dev\ml_mosaic.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\dev\ml_mosaic.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\dev\ml_mosaic.py
)

[mlmodels\model_dev\dev\mytest.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\dev\mytest.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\dev\mytest.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\dev\mytest.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\dev\mytest.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\dev\mytest.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_dev\dev\mytest.py
)

[mlmodels\model_flow\__init__.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\__init__.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\__init__.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\__init__.py
)

[mlmodels\model_flow\dev\mlflow_run.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\dev\mlflow_run.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\dev\mlflow_run.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\dev\mlflow_run.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\dev\mlflow_run.py
)

[cli_load_arguments(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\dev\mlflow_run.py
)

[log_scalar(name, value, step,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\dev\mlflow_run.py
)

[mlflow_add(args,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\dev\mlflow_run.py
)

[session_init(args,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\dev\mlflow_run.py
)

[tfboard_add_weights(step,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\dev\mlflow_run.py
)

[tfboard_writer_create(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\dev\mlflow_run.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\dev\mlflow_run.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_flow\dev\mlflow_run.py
)

[mlmodels\model_gluon\gluon_automl.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_automl.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_automl.py
)

[Model.__init__(self,  model_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_automl.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_automl.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_automl.py
)

[_config_process(config,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_automl.py
)

[get_params( choice="", data_path="dataset/", config_mode="test",  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_automl.py
)

[path_setup( out_folder="", sublevel=0, data_path="dataset/",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_automl.py
)

[test( data_path="dataset/", pars_choice="json",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_automl.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_automl.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_automl.py
)

[mlmodels\model_gluon\gluon_deepar.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_deepar.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_deepar.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_deepar.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_deepar.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_deepar.py
)

[get_params( choice="", data_path="dataset/timeseries/", config_mode="test",  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_deepar.py
)

[test( data_path="dataset/", choice="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_deepar.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_deepar.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_deepar.py
)

[mlmodels\model_gluon\gluon_ffn.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_ffn.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_ffn.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_ffn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_ffn.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_ffn.py
)

[get_params( choice="", data_path="dataset/timeseries/", config_mode="test",  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_ffn.py
)

[test( data_path="dataset/", choice="test01",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_ffn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_ffn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_ffn.py
)

[mlmodels\model_gluon\gluon_prophet.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_prophet.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_prophet.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_prophet.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_prophet.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_prophet.py
)

[get_params( choice="", data_path="dataset/", config_mode="test",  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_prophet.py
)

[test( data_path="dataset/", choice="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_prophet.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_prophet.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\gluon_prophet.py
)

[mlmodels\model_gluon\util.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[Model_empty.__init__(self,  model_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[_config_process(data_path,  config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[fit(model,  sess=None, data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[get_dataset(data_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[load(path,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[metrics(ypred, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[plot_predict(item_metrics,  out_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[plot_prob_forecasts(ypred,  out_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[save(model, path,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util.py
)

[mlmodels\model_gluon\util_autogluon.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[Model_empty.__init__(self,  model_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[_get_dataset_from_aws(  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[fit(model,  data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[get_dataset(  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[import_data_fromfile(  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[load(path,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[log( n=0, m=1,  *s)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[metrics(model, ypred, ytrue, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[os_package_root_path(filepath,  sublevel=0, path_add="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[predict(model, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[save(model, out_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\util_autogluon.py
)

[mlmodels\model_gluon\__init__.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\__init__.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\__init__.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_gluon\__init__.py
)

[mlmodels\model_keras\01_deepctr.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[_config_process(config,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[_preprocess_criteo(df,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[_preprocess_movielens(df,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[config_load(data_path, file_default, config_mode,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[fit(model,  session=None, compute_pars=None, data_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[get_dataset( data_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[get_params( choice="", data_path="dataset/", config_mode="test",  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[metrics(ypred,  ytrue=None, session=None, compute_pars=None, data_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[path_setup( out_folder="", sublevel=0, data_path="dataset/",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[predict(model,  session=None, compute_pars=None, data_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[reset_model(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[test( data_path="dataset/", pars_choice=0,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\01_deepctr.py
)

[mlmodels\model_keras\02_cnn.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[Model.__init__(self,  model_pars=None, compute_pars=None, data_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[fit(model,  data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[get_dataset(data_params,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[get_params( choice=0, data_path="dataset/",  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[log( n=0, m=1,  *s)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[metrics(ypred, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[os_package_root_path(filepath,  sublevel=0, path_add="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[predict(model, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[test( data_path="dataset/",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[test2( data_path="dataset/", out_path="keras/keras.png", reset=True,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\02_cnn.py
)

[mlmodels\model_keras\armdn.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[fit( model=None, data_pars={}, compute_pars={}, out_pars={},  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[get_dataset( data_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[get_dataset(data_params,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[get_params( choice=0, data_path="dataset/",  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[load( load_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[log( n=0, m=1,  *s)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[os_package_root_path(filepath,  sublevel=0, path_add="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[predict( model=None, model_pars=None, data_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[save( model=None, session=None, save_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[test( data_path="dataset/",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\armdn.py
)

[mlmodels\model_keras\charcnn.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[get_dataset( data_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[get_params( param_pars={},  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[load( load_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[predict(model,  sess=None, data_pars=None, out_pars=None, compute_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[reset_model(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[save( model=None, session=None, save_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[test( data_path="dataset/", pars_choice="json", config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn.py
)

[mlmodels\model_keras\charcnn_zhang.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[fit(model,  data_pars={}, compute_pars={}, out_pars={},  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[fit_metrics(model,  data_pars={}, compute_pars={}, out_pars={},  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[get_dataset( data_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[get_params( param_pars={},  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[load( load_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[predict(model,  sess=None, data_pars={}, out_pars={}, compute_pars={},  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[reset_model(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[save( model=None, session=None, save_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[test( data_path="dataset/", pars_choice="json", config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\charcnn_zhang.py
)

[mlmodels\model_keras\namentity_crm_bilstm.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[_preprocess_test(data_pars,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[get_dataset(data_pars,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[get_params( param_pars={},  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[load(load_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[predict(model,  sess=None, data_pars=None, out_pars=None, compute_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[reset_model(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[save( model=None, session=None, save_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[test( data_path="dataset/", pars_choice="json", config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\namentity_crm_bilstm.py
)

[mlmodels\model_keras\preprocess.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\preprocess.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\preprocess.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\preprocess.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\preprocess.py
)

[_preprocess_criteo(df,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\preprocess.py
)

[_preprocess_movielens(df,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\preprocess.py
)

[_preprocess_none(df,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\preprocess.py
)

[get_dataset(  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\preprocess.py
)

[log( n=0, m=1,  *s)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\preprocess.py
)

[os_package_root_path(filepath,  sublevel=0, path_add="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\preprocess.py
)

[test( data_path="dataset/", pars_choice=0,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\preprocess.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\preprocess.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\preprocess.py
)

[mlmodels\model_keras\textcnn.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[get_dataset( data_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[get_params( param_pars={},  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[load( load_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[predict(model,  sess=None, data_pars=None, out_pars=None, compute_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[reset_model(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[save( model=None, session=None, save_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[test( data_path="dataset/", pars_choice="json", config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textcnn.py
)

[mlmodels\model_keras\textvae.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[CustomVariationalLayer.__init__(self,   **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[CustomVariationalLayer.call(self, inputs,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[CustomVariationalLayer.vae_loss(self, x, x_decoded_mean,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[get_dataset( data_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[get_params( param_pars={},  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[load( load_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[reset_model(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[save( model=None, session=None, save_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[test( data_path="dataset/", pars_choice="json", config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\textvae.py
)

[mlmodels\model_keras\util.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[Model_empty.__init__(self,  model_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[_config_process(data_path,  config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[fit(model,  data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[get_dataset(  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[load(path,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[log( n=0, m=1,  *s)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[metrics(ypred, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[os_package_root_path(filepath,  sublevel=0, path_add="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[predict(model, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[save(model, path,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\util.py
)

[mlmodels\model_keras\__init__.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\__init__.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\__init__.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_keras\__init__.py
)

[mlmodels\model_rank\__init__.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\__init__.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\__init__.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\__init__.py
)

[mlmodels\model_rank\dev\LambdaRank.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\LambdaRank.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\LambdaRank.py
)

[LambdaRank.__init__(self, net_structures,  leaky_relu=False, sigma=1.0, double_precision=False,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\LambdaRank.py
)

[LambdaRank.dump_param(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\LambdaRank.py
)

[LambdaRank.forward(self, input1,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\LambdaRank.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\LambdaRank.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\LambdaRank.py
)

[train( start_epoch=0, additional_epoch=100, lr=0.0001, optim="adam", leaky_relu=False, ndcg_gain_in_train="exp2", sigma=1.0, double_precision=False, standardize=False, small_dataset=False, debug=False,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\LambdaRank.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\LambdaRank.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\LambdaRank.py
)

[mlmodels\model_rank\dev\load_mslr.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[DataLoader.__init__(self, path,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[DataLoader._load_mslr(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[DataLoader._parse_feature_and_label(self, df,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[DataLoader.apply_scaler(self, scaler,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[DataLoader.generate_batch_per_query(self,  df=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[DataLoader.generate_query_batch(self, df,  batchsize=100000,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[DataLoader.generate_query_pair_batch(self,  df=None, batchsize=2000,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[DataLoader.generate_query_pairs(self, df, qid,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[DataLoader.get_num_pairs(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[DataLoader.get_num_sessions(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[DataLoader.load(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[DataLoader.train_scaler_and_transform(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[get_time(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\load_mslr.py
)

[mlmodels\model_rank\dev\metrics.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\metrics.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\metrics.py
)

[DCG.__init__(self,  k=10, gain_type='exp2',  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\metrics.py
)

[DCG._get_discount(self, k,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\metrics.py
)

[DCG._get_gain(self, targets,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\metrics.py
)

[DCG._make_discount(n,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\metrics.py
)

[DCG.evaluate(self, targets,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\metrics.py
)

[NDCG.__init__(self,  k=10, gain_type='exp2',  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\metrics.py
)

[NDCG.evaluate(self, targets,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\metrics.py
)

[NDCG.maxDCG(self, targets,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\metrics.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\metrics.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\metrics.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\metrics.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\metrics.py
)

[mlmodels\model_rank\dev\RankNet.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[RankNet.__init__(self, net_structures,  double_precision=False,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[RankNet.dump_param(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[RankNet.forward(self, input1,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[RankNetPairs.__init__(self, net_structures,  double_precision=False,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[RankNetPairs.forward(self, input1, input2,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[baseline_pairwise_training_loop(epoch, net, loss_func, optimizer, train_loader,  batch_size=100000, precision=torch.float32, device="cpu", debug=False,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[eval_model(inference_model, device, df_valid, valid_loader,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[factorized_training_loop(epoch, net, loss_func, optimizer, train_loader,  batch_size=200, sigma=1.0, training_algo=SUM_SESSION, precision=torch.float32, device="cpu", debug=False,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[get_train_inference_net(train_algo, num_features, start_epoch, double_precision,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[load_from_ckpt(ckpt_file, epoch, model,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[train_rank_net( start_epoch=0, additional_epoch=100, lr=0.0001, optim="adam", train_algo=SUM_SESSION, double_precision=False, standardize=False, small_dataset=False, debug=False,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\RankNet.py
)

[mlmodels\model_rank\dev\utils.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[eval_cross_entropy_loss(model, device, loader,  phase="Eval", sigma=1.0,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[eval_ndcg_at_k(inference_model, device, df_valid, valid_loader, batch_size, k_list,  phase="Eval",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[get_args_parser(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[get_ckptdir(net_name, net_structure,  sigma=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[get_device(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[init_weights(m,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[load_train_vali_data(data_fold,  small_dataset=False,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[save_to_ckpt(ckpt_file, epoch, model, optimizer, lr_scheduler,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[str2bool(v,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_rank\dev\utils.py
)

[mlmodels\model_sklearn\model_lightgbm.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[get_dataset( data_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[get_params( param_pars={},  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[load( load_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[reset_model(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[save( model=None, session=None, save_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[test( data_path="dataset/", pars_choice="json", config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_lightgbm.py
)

[mlmodels\model_sklearn\model_sklearn.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[get_dataset( data_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[get_params( param_pars={},  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[load( load_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[reset_model(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[save( model=None, session=None, save_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[test( data_path="dataset/", pars_choice="json", config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\model_sklearn.py
)

[mlmodels\model_sklearn\__init__.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\__init__.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\__init__.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_sklearn\__init__.py
)

[mlmodels\model_tch\02_mlp.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\02_mlp.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\02_mlp.py
)

[Model.__init__(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\02_mlp.py
)

[Model.forward(self, x,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\02_mlp.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\02_mlp.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\02_mlp.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\02_mlp.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\02_mlp.py
)

[mlmodels\model_tch\03_nbeats.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[data_generator(x_full, y_full, bs,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[fit_simple(net, optimiser, data_generator, on_save_callback, device, data_pars, out_pars,  max_grad_steps=500,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[get_dataset(  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[get_params(param_pars,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[load(load_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[load_checkpoint(model, optimiser,  CHECKPOINT_NAME='nbeats-fiting-checkpoint.th',  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[plot(net, x, target, backcast_length, forecast_length, grad_step,  out_path="./",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[plot_model(net, x, target, grad_step, data_pars,  disable_plot=False,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[plot_predict(x_test, y_test, p, data_pars, compute_pars, out_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[predict(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[save(model, session, save_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[save_checkpoint(model, optimiser, grad_step,  CHECKPOINT_NAME="mycheckpoint",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[test( data_path="dataset/milk.csv",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\03_nbeats.py
)

[mlmodels\model_tch\pplm.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\pplm.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\pplm.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\pplm.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\pplm.py
)

[fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\pplm.py
)

[generate(cond_text, bag_of_words,  discrim=None, class_label=-1,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\pplm.py
)

[get_dataset( data_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\pplm.py
)

[get_params( param_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\pplm.py
)

[path_setup( out_folder="", sublevel=0, data_path="dataset/",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\pplm.py
)

[predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\pplm.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\pplm.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\pplm.py
)

[mlmodels\model_tch\textcnn.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[TextCNN.__init__(self,  model_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[TextCNN.forward(self, x,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[TextCNN.rebuild_embed(self, vocab_built,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[TextCNN.tokenizer(text,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[_get_device(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[_train(m, device, train_itr, optimizer, epoch, max_epoch,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[_valid(m, device, test_itr,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[clean_str(string,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[create_data_iterator(tr_batch_size, val_batch_size, tabular_train, tabular_valid, d,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[create_tabular_dataset(path_train, path_valid,  lang='en', pretrained_emb='glove.6B.300d',  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[fit(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[get_config_file(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[get_data_file(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[get_dataset( data_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[get_params( param_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[load(path,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[metric(model,  data_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[predict(model,  data_pars=None, compute_pars=None, out_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[save(model, path,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[split_train_valid(path_data, path_train, path_valid,  frac=0.7,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[test(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\textcnn.py
)

[mlmodels\model_tch\torchhub.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None, out_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[_get_device(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[_train(m, device, train_itr, criterion, optimizer, epoch, max_epoch,  imax=1,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[_valid(m, device, test_itr, criterion,  imax=1,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[get_config_file(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[get_dataset( data_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[get_dataset_mnist_torch(data_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[get_params( param_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[load(load_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[predict(model,  session=None, data_pars=None, compute_pars=None, out_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[save(model, session, save_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[test( data_path="dataset/", pars_choice="json", config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\torchhub.py
)

[mlmodels\model_tch\transformer_classifier.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[_preprocess_XXXX(df,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[fit(train_dataset, model, tokenizer,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[fit_metrics(model, tokenizer, model_pars, data_pars, out_pars, compute_pars,  prefix="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[get_dataset(task, tokenizer,  evaluate=False,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[get_eval_report(labels, preds,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[get_mismatched(labels, preds,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[get_params( param_pars={},  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[load( load_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[load_and_cache_examples(task, tokenizer,  evaluate=False,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[metrics(task_name, preds, labels,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[reset_model(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[save( model=None, session=None, save_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[test(data_path, model_pars, data_pars, compute_pars, out_pars,  pars_choice=0,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier.py
)

[mlmodels\model_tch\transformer_classifier2.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[Model_empty.__init__(self,  model_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[_preprocess_XXXX(df,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[evaluate(model, tokenizer,  prefix="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[fit(model,  data_pars=None, model_pars={}, compute_pars=None, out_pars=None,  *args, **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[get_dataset( data_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[get_eval_report(labels, preds,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[get_mismatched(labels, preds,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[get_params( choice=0, data_path="dataset/",  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[load( out_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[metrics(task_name, preds, labels,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[metrics_evaluate(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[path_setup( out_folder="", sublevel=0, data_path="dataset/",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[predict(model,  sess=None, data_pars=None, out_pars=None, compute_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[reset_model(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[save(model, out_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[test( data_path="dataset/", pars_choice=0,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_classifier2.py
)

[mlmodels\model_tch\transformer_sentence.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[fit(model,  data_pars=None, model_pars={}, compute_pars=None, out_pars=None,  *args, **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[fit_metrics(model,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[get_dataset( data_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[get_params(param_pars,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[load( out_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[predict(model,  sess=None, data_pars=None, out_pars=None, compute_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[reset_model(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[save(model, out_pars,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[test( data_path="dataset/", pars_choice="test01",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\transformer_sentence.py
)

[mlmodels\model_tch\util_data.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_data.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_data.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_data.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_data.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_data.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_data.py
)

[mlmodels\model_tch\util_transformer.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[BinaryProcessor._create_examples(self, lines, set_type,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[BinaryProcessor.get_dev_examples(self, data_dir,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[BinaryProcessor.get_labels(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[BinaryProcessor.get_train_examples(self, data_dir,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[DataProcessor._read_tsv(cls, input_file,  quotechar=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[DataProcessor.get_dev_examples(self, data_dir,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[DataProcessor.get_labels(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[DataProcessor.get_train_examples(self, data_dir,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[InputExample.__init__(self, guid, text_a,  text_b=None, label=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[InputFeatures.__init__(self, input_ids, input_mask, segment_ids, label_id,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[_truncate_seq_pair(tokens_a, tokens_b, max_length,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[convert_example_to_feature(example_row,  pad_token=0, sequence_a_segment_id=0, sequence_b_segment_id=1, cls_token_segment_id=1, pad_token_segment_id=0, mask_padding_with_zero=True, sep_token_extra=False,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode,  cls_token_at_end=False, sep_token_extra=False, pad_on_left=False, cls_token='[CLS]', sep_token='[SEP]', pad_token=0, sequence_a_segment_id=0, sequence_b_segment_id=1, cls_token_segment_id=1, pad_token_segment_id=0, mask_padding_with_zero=True, process_count=cpu_count(),  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\util_transformer.py
)

[mlmodels\model_tch\__init__.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\__init__.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\__init__.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tch\__init__.py
)

[mlmodels\model_tf\1_lstm.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kwarg)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[fit_metrics(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[get_dataset( data_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[get_params( param_pars={},  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[load( load_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[metrics(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None, get_hidden_state=False, init_value=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[reset_model(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[save(model,  session=None, save_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[test( data_path="dataset/", pars_choice="test01", config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\1_lstm.py
)

[mlmodels\model_tf\util.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\util.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\util.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\util.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\util.py
)

[batch_gather(values, indices,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\util.py
)

[batch_invert_permutation(permutations,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\util.py
)

[one_hot(length, index,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\util.py
)

[os_file_path(data_path,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\util.py
)

[os_module_path(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\util.py
)

[os_package_root_path(filepath,  sublevel=0, path_add="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\util.py
)

[set_root_dir(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\util.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\util.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\util.py
)

[mlmodels\model_tf\__init__.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\__init__.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\__init__.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\__init__.py
)

[mlmodels\model_tf\rl\0_template_rl.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[Agent.__init__(self, history, do_action,  params={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[Agent.discount_rewards(self, r,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[Agent.get_predicted_action(self, sequence,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[Agent.get_state(self, t,  state=None, history=None, reward=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[Agent.predict_action(self, inputs,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[Agent.run_sequence(self, history, do_action, params,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[Agent.train(self,  n_iters=1, n_log_freq=1, state_initial=None, reward_initial=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[Model.__init__(self, history,  params={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[do_action_example(action_dict,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[fit(model, df, do_action,  state_initial=None, reward_initial=None, params=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[predict(model, sess, df,  do_action=None, params=params,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[val(x, y,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\0_template_rl.py
)

[mlmodels\model_tf\rl\1.turtle-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\1.turtle-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\1.turtle-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\1.turtle-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\1.turtle-agent.py
)

[buy_stock(real_movement, signal,  initial_money=10000, max_buy=20, max_sell=20,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\1.turtle-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\1.turtle-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\1.turtle-agent.py
)

[mlmodels\model_tf\rl\10.duel-q-learning-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\10.duel-q-learning-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\10.duel-q-learning-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip, batch_size,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\10.duel-q-learning-agent.py
)

[Agent.act(self, state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\10.duel-q-learning-agent.py
)

[Agent.buy(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\10.duel-q-learning-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\10.duel-q-learning-agent.py
)

[Agent.replay(self, batch_size,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\10.duel-q-learning-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\10.duel-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\10.duel-q-learning-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\10.duel-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\10.duel-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\10.duel-q-learning-agent.py
)

[mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[Agent._assign(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[Agent._construct_memories(self, replay,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[Agent._memorize(self, state, action, reward, new_state, done,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[Agent._select_action(self, state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[Agent.buy(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[Agent.get_predicted_action(self, sequence,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[Agent.predict(self, inputs,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[Model.__init__(self, input_size, output_size, layer_size, learning_rate,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
)

[mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
)

[Agent._construct_memories(self, replay,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
)

[Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
)

[Agent.buy(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
)

[mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[Agent._assign(self, from_name, to_name,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[Agent._construct_memories(self, replay,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[Agent._select_action(self, state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[Agent.buy(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[Model.__init__(self, input_size, output_size, layer_size, learning_rate, name,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
)

[mlmodels\model_tf\rl\14.actor-critic-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[Actor.__init__(self, name, input_size, output_size, size_layer,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[Agent._assign(self, from_name, to_name,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[Agent._construct_memories_and_train(self, replay,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[Agent._memorize(self, state, action, reward, new_state, dead,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[Agent._select_action(self, state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[Agent.buy(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\14.actor-critic-agent.py
)

[mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[Actor.__init__(self, name, input_size, output_size, size_layer,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[Agent._assign(self, from_name, to_name,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[Agent._construct_memories_and_train(self, replay,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[Agent._memorize(self, state, action, reward, new_state, dead,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[Agent._select_action(self, state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[Agent.buy(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
)

[mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[Actor.__init__(self, name, input_size, output_size, size_layer,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[Agent._assign(self, from_name, to_name,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[Agent._construct_memories_and_train(self, replay,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[Agent._select_action(self, state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[Agent.buy(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
)

[mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[Actor.__init__(self, name, input_size, output_size, size_layer,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[Agent._assign(self, from_name, to_name,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[Agent._construct_memories_and_train(self, replay,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[Agent._select_action(self, state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[Agent.buy(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
)

[mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[Agent._construct_memories(self, replay,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[Agent._memorize(self, state, action, reward, new_state, done,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[Agent._select_action(self, state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[Agent.buy(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[Agent.get_predicted_action(self, sequence,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[Agent.predict(self, inputs,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
)

[mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
)

[Agent._construct_memories(self, replay,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
)

[Agent._memorize(self, state, action, reward, new_state, done, rnn_state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
)

[Agent.buy(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
)

[mlmodels\model_tf\rl\2.moving-average-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\2.moving-average-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\2.moving-average-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\2.moving-average-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\2.moving-average-agent.py
)

[buy_stock(real_movement, signal,  initial_money=10000, max_buy=20, max_sell=20,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\2.moving-average-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\2.moving-average-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\2.moving-average-agent.py
)

[mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[Agent._construct_memories(self, replay,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[Agent._memorize(self, state, action, reward, new_state, done,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[Agent._select_action(self, state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[Agent.buy(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[Agent.get_predicted_action(self, sequence,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[Agent.predict(self, inputs,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
)

[mlmodels\model_tf\rl\21.neuro-evolution-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[NeuroEvolution.__init__(self, population_size, mutation_rate, model_generator, state_size, window_size, trend, skip, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[NeuroEvolution._initialize_population(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[NeuroEvolution.act(self, p, state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[NeuroEvolution.buy(self, individual,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[NeuroEvolution.calculate_fitness(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[NeuroEvolution.crossover(self, parent1, parent2,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[NeuroEvolution.evolve(self,  generations=20, checkpoint=5,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[NeuroEvolution.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[NeuroEvolution.inherit_weights(self, parent, child,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[NeuroEvolution.mutate(self, individual,  scale=1.0,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[neuralnetwork.__init__(self, id_,  hidden_size=128,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[feed_forward(X, nets,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[relu(X,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[softmax(X,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\21.neuro-evolution-agent.py
)

[mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[NeuroEvolution.__init__(self, population_size, mutation_rate, model_generator, state_size, window_size, trend, skip, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[NeuroEvolution._initialize_population(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[NeuroEvolution._memorize(self, q, i, limit,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[NeuroEvolution.act(self, p, state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[NeuroEvolution.buy(self, individual,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[NeuroEvolution.calculate_fitness(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[NeuroEvolution.crossover(self, parent1, parent2,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[NeuroEvolution.evaluate(self, individual, backlog, pop,  k=4,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[NeuroEvolution.evolve(self,  generations=20, checkpoint=5,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[NeuroEvolution.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[NeuroEvolution.inherit_weights(self, parent, child,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[NeuroEvolution.mutate(self, individual,  scale=1.0,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[neuralnetwork.__init__(self, id_,  hidden_size=128,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[feed_forward(X, nets,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[relu(X,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[softmax(X,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
)

[mlmodels\model_tf\rl\3.signal-rolling-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\3.signal-rolling-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\3.signal-rolling-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\3.signal-rolling-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\3.signal-rolling-agent.py
)

[buy_stock(real_movement,  delay=5, initial_state=1, initial_money=10000, max_buy=20, max_sell=20,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\3.signal-rolling-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\3.signal-rolling-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\3.signal-rolling-agent.py
)

[mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[Agent.buy(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[Agent.discount_rewards(self, r,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[Agent.get_predicted_action(self, sequence,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[Agent.get_state(self, t,  reward_state=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[Agent.predict(self, inputs,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[Agent.predict_sequence(self, trend_input, do_action,  param=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[Model.__init__(self, state_size, window_size, trend, skip, iterations, initial_reward,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[do_action_example(action_dict,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[fit(model, df, do_action,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[predict(model, sess, df, do_action,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[test( filename='dataset/GOOG-year.csv',  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
)

[mlmodels\model_tf\rl\4_policy-gradient-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[Agent.discount_rewards(self, r,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[Agent.get_predicted_action(self, sequence,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[Agent.get_state(self, t,  reward_state=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[Agent.predict(self, inputs,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[Agent.predict_sequence(self, pars,  trend_history=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[Model.__init__(self, state_size, window_size, trend, skip, iterations, initial_reward,  checkpoint=10,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[fit(model, dftrain,  params={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[predict(model, sess, dftest,  params={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[test( filename='dataset/GOOG-year.csv',  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\4_policy-gradient-agent.py
)

[mlmodels\model_tf\rl\5_q-learning-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip, batch_size,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[Agent.act(self, state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[Agent.predict_sequence(self, pars,  trend_history=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[Agent.replay(self, batch_size,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[Model.__init__(self, state_size, window_size, trend, skip, iterations, initial_reward,  checkpoint=10,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[fit(model, dftrain,  params={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[predict(model, sess, dftest,  params={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[test( filename='../dataset/GOOG-year.csv',  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\5_q-learning-agent.py
)

[mlmodels\model_tf\rl\6_evolution-strategy-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[Agent.__init__(self, model, window_size, trend, skip, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[Agent.act(self, sequence,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[Agent.fit(self, iterations, checkpoint,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[Agent.get_reward(self, weights,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[Agent.run_sequence(self, df_test,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[Deep_Evolution_Strategy.__init__(self, weights, reward_function, population_size, sigma, learning_rate,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[Deep_Evolution_Strategy._get_weight_from_population(self, weights, population,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[Deep_Evolution_Strategy.get_weights(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[Deep_Evolution_Strategy.train(self,  epoch=100, print_every=1,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[Model.__init__(self, input_size, layer_size, output_size, window_size, skip, initial_money,  iterations=500, checkpoint=10,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[Model.get_weights(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[Model.predict(self, inputs,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[Model.set_weights(self, weights,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[fit(model, dftrain,  params={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[get_imports(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[predict(model, sess, dftest,  params={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[test( filename='../dataset/GOOG-year.csv',  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\6_evolution-strategy-agent.py
)

[mlmodels\model_tf\rl\7.double-q-learning-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[Agent._assign(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[Agent._construct_memories(self, replay,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[Agent._memorize(self, state, action, reward, new_state, done,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[Agent._select_action(self, state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[Agent.get_predicted_action(self, sequence,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[Agent.predict(self, inputs,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[Agent.run_sequence(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[Model.__init__(self, window_size, trend, skip, iterations, initial_reward,  checkpoint=10,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[QModel.__init__(self, input_size, output_size, layer_size, learning_rate,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[fit(model, dftrain,  params={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[predict(model, sess, dftest,  params={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[test( filename='../dataset/GOOG-year.csv',  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\7.double-q-learning-agent.py
)

[mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
)

[Agent._construct_memories(self, replay,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
)

[Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
)

[Agent.buy(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
)

[mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[Agent.__init__(self, state_size, window_size, trend, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[Agent._assign(self, from_name, to_name,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[Agent._construct_memories(self, replay,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[Agent._select_action(self, state,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[Agent.buy(self, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[Agent.get_state(self, t,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[Agent.train(self, iterations, checkpoint, initial_money,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[Model.__init__(self, input_size, output_size, layer_size, learning_rate, name,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
)

[mlmodels\model_tf\rl\updated-NES-google.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[Agent.__init__(self, model, money, max_buy, max_sell, close, window_size, skip,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[Agent.act(self, sequence,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[Agent.buy(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[Agent.fit(self, iterations, checkpoint,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[Agent.get_reward(self, weights,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[Deep_Evolution_Strategy.__init__(self, weights, reward_function, population_size, sigma, learning_rate,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[Deep_Evolution_Strategy._get_weight_from_population(self, weights, population,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[Deep_Evolution_Strategy.get_weights(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[Deep_Evolution_Strategy.train(self,  epoch=100, print_every=1,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[Model.__init__(self, input_size, layer_size, output_size,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[Model.get_weights(self,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[Model.predict(self, inputs,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[Model.set_weights(self, weights,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[act(model, sequence,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[f(w,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[get_state(data, t, n,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\updated-NES-google.py
)

[mlmodels\model_tf\rl\__init__.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\__init__.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\__init__.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\model_tf\rl\__init__.py
)

[mlmodels\preprocess\__init__.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\preprocess\__init__.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\preprocess\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\preprocess\__init__.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\preprocess\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\preprocess\__init__.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\preprocess\__init__.py
)

[mlmodels\template\00_template_keras.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[Model_empty.__init__(self,  model_pars=None, compute_pars=None,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[_preprocess_XXXX(df,   **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[fit(model,  session=None, data_pars=None, model_pars=None, compute_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[get_dataset(  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[get_params( choice=0, data_path="dataset/",  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[load(path,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[log( n=0, m=1,  *s)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[metrics(ypred, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[os_package_root_path(filepath,  sublevel=0, path_add="",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[path_setup( out_folder="", sublevel=0, data_path="dataset/",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[predict(model, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[reset_model(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[save(model, path,   )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[test( data_path="dataset/", pars_choice=0,  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\00_template_keras.py
)

[mlmodels\template\model_xxx.py
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)

[----------------methods----------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)

[---------------functions---------------
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)

[fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)

[fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)

[get_dataset( data_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)

[get_params( param_pars={},  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)

[load( load_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)

[predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kw)
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)

[reset_model(  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)

[save( model=None, session=None, save_pars={},  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)

[test( data_path="dataset/", pars_choice="json", config_mode="test",  )
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)

[
](https://github.com/arita37/mlmodels/tree/dev/mlmodels\template\model_xxx.py
)


    </body>
    </html>

    