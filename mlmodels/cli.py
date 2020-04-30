import click
from mlmodels import models
from mlmodels.util import log

@click.group()
def main():
    pass

@main.command()
@click.option('-f', '--folder', help="Use option to mention folder explictly")
def list(folder=None):
    models.config_model_list(folder)

@main.command()
@click.option('-f', '--folder', help="Use option to mention folder explictly")
def testall():
    models.test_all(folder)

@main.command()
@click.argument('model_name')
@click.option('-p', '--params', help="Enter Model Params")
def test(model_name, params):
    # Name as argument as must entered by user,
    # params are optional. Would take default params
    # if user didn;t mention explictly
    models.test(model_name)
    # both of these are same thus, test are running two times.
    # one is implemented in models.py and one is implemented
    # individually in each model.
    param_pars = {"choice": "test01", "data_path": "", "config_mode": "test"}
    models.test_module(model_name, param_pars=param_pars)




@main.command()
@click.argument('model_name')
@click.option('-cf', '--config_file', help="Enter Path to config file")
@click.option('cm', '--config_mode', help="Enter Config Mode: test/prod/uat", default="test")
@click.option('-sf', '--save_folder', help="Model Save Location")
def fit(model_name, config_file, config_mode, save_folder):
    from mlmodels.models import fit_cli
    fit_cli(model_name, config_file, config_mode, save_folder)




@main.command()
@click.argument('model_name')
@click.option('-cf', '--config_file', help="Enter Path to config file")
@click.option('cm', '--config_mode', help="Enter Config Mode: test/prod/uat", default="test")
@click.option('-sf', '--save_folder', help="Model Save Location")
def predict():
    from mlmodels.models import predict_cli
    predict_cli(model_name, config_file, config_mode, save_folder)



@main.command()
@click.argument('model_name')
@click.option('-sf', '--save_folder', help="Folder Path to save configuration")
def generate_config(model_name, save_folder):
    log(arg.save_folder)
    models.config_generate_json(model_name, to_path=save_folder)


if __name__ == "__main__":
    main()