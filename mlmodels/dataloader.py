import os
import sys
import inspect
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import sklearn
import keras
from sklearn.model_selection import train_test_split
from cli_code.cli_download import Downloader
from collections.abc import MutableMapping
import json
from importlib import import_module
import pickle


class DataLoaderError(Exception):
    pass


class MissingLocationKeyError(DataLoaderError):
    def __init__(self):
        print('Location key missing from the input dictionary.')


class UndeterminableLocationTypeError(DataLoaderError):
    def __init__(self):
        print('Location type cannot be inferred.')


class UnknownLocationTypeError(DataLoaderError):
    def __init__(self, location_type):
        print(f"Location type '{location_type}' is unknown.")


class NonfileURLError(DataLoaderError):
    def __init__(self):
        print(f'URL must point to a file.')


class UndeterminableDataLoaderError(DataLoaderError):
    def __init__(self):
        print(
            f"""Loader function to be used was not provided and could not be
             automatically inferred from file type."""
        )


class NonIntegerBatchSizeError(DataLoaderError):
    def __init__(self):
        print(f'Provided batch size cannot be interpreted as an integer.')


class InvalidDataLoaderFunctionError(DataLoaderError):
    def __init__(self, loader):
        print(f'Invalid data loader function \'{loader}\ specified.')


class NumpyGeneratorError(DataLoaderError):
    def __init__(self):
        print(f'Loading Numpy binaries as generators is unsupported.')


class MissingDataPreprocessorError(DataLoaderError):
    def __init__(self):
        print(f'data_preprocessor is missing in preprocessor.')


class InvalidDataPreprocessorParameterError(DataLoaderError):
    def __init__(self, parameter):
        print(f'Could not evaluate data_preprocessor parameter {parameter}.')


class InvalidEncoderParameterError(DataLoaderError):
    def __init__(self, parameter):
        print(f'Could not evaluate encoder parameter {parameter}.')


class InvalidDataPreprocessorError(DataLoaderError):
    def __init__(self, preprocessor):
        print(f'Could not evaluate data_preprocessor \'{preprocessor}\'.')


class InvalidEncoderError(DataLoaderError):
    def __init__(self, preprocessor):
        print(f'Could not evaluate encoder \'{preprocessor}\'.')


class NonCallableDataPreprocessorError(DataLoaderError):
    def __init__(self, preprocessor):
        print(f'\'{preprocessor}\' is not callable.')


class NonCallableEncoderError(DataLoaderError):
    def __init__(self, preprocessor):
        print(f'\'{preprocessor}\' is not callable.')


class EncoderMissingIndexError(DataLoaderError):
    def __init__(self, encoder_pars):
        print(f'\'{encoder_pars}\' is missing the index parameter.')


class EncoderMissingEncoderError(DataLoaderError):
    def __init__(self, encoder_pars):
        print(f'\'{encoder_pars}\' is missing the encoder parameter.')


class OutputShapeError(DataLoaderError):
    def __init__(self, specified, actual):
        print(
            f'''Specified output shape {specified} does not match actual output
            shape {actual}'''
        )


class PreprocssingOutputDict(dict):
    def __init__(self, *args, **kwargs):
        super(PreprocssingOutputDict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        if isinstance(key, list):
            return (super(PreprocssingOutputDict, self).__getitem__(k)
                    for k in key)
        else:
            return super(PreprocssingOutputDict, self).__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, list):
            (super(PreprocssingOutputDict, self).__setitem__(k, v)
             for k, v in zip(key, value))
        else:
            super(PreprocssingOutputDict, self).__setitem__(key, value)

    def __str__(self):
        return super(PreprocssingOutputDict, self).__str__()

    def __repr__(self):
        return super(PreprocssingOutputDict, self).__repr__()

    def values(self):
        return super(PreprocssingOutputDict, self).values()


def load_function(f):
    try:
        return eval(f)
    except NameError:
        import_module(f.rsplit('.', 1)[0])
        return eval(f)


class AbstractDataLoader:

    default_loaders = {
        '.csv': 'pd.read_csv',
        '.txt': 'lambda f: open(f).read()',
        '.npy': 'np.load',
        '.npz': 'np.load',
        '.pkl': 'lambda f: pickle.load(open(f,'
        r'))'
    }

    def __init__(self, input_pars, loader, preprocessor, output):
        self.preprocessor = preprocessor
        self.state_preprocessors = []
        self._interpret_input_pars(input_pars)
        loaded_data = self._load_data(loader)
        processed_data = self._interpret_processor(preprocessor, loaded_data)
        X_list = []
        if self.X_cols is not None:
            for x in self.X_cols:
                if isinstance(x, str):
                    X_list.append(processed_data[x])
                else:
                    try:
                        X_list.append(processed_data[:, x])
                    except:
                        X_list.append(processed_data[x])
        y_list = []
        if self.y_cols is not None:
            for y in self.y_cols:
                if isinstance(x, str):
                    y_list.append(processed_data[y])
                else:
                    try:
                        y_list.append(processed_data[:, y])
                    except:
                        y_list.append(processed_data[y])
        misc_list = []
        if self.misc_cols is not None:
            for m in self.misc_cols:
                if isinstance(x, str):
                    misc_list.append(processed_data[m])
                else:
                    try:
                        misc_list.append(processed_data[:, m])
                    except:
                        misc_list.append(processed_data[m])
        if self.test_size > 0 and not self.generator and len(
                X_list) > 0 and len(y_list) > 0:
            processed_data = train_test_split(
                *(X_list + y_list), test_size=self.test_size) + misc_list
        if isinstance(processed_data, PreprocssingOutputDict):
            processed_data = list(processed_data.values())
        if isinstance(processed_data, list) and len(processed_data) == 1:
            processed_data = processed_data[0]
        print([x.shape for x in processed_data])
        self.intermediate_output = processed_data
        self._interpret_output(output)

    def _interpret_input_pars(self, input_pars):
        try:
            location = input_pars['location']
        except KeyError:
            raise MissingLocationKeyError()

        location_type = input_pars.get('location_type', None)
        if location_type is None:
            if os.path.isfile(location):
                location_type = 'file'
            if os.path.isdir(location):
                location_type = 'dir'
            if urlparse(location).scheme != '':
                location_type = 'url'
                download_path = input_pars.get('download_path', './')
            if location_type == 'dropbox':
                dropbox_download(location)
                location_type = 'file'
            if location_type is None:
                raise UndeterminableLocationTypeError(location_type)
        elif location_type != 'file' and location_type != 'dir' \
                and location_type != 'url':
                raise UnknownLocationTypeError()

        file_type = input_pars.get('file_type', None)
        if file_type is None:
            if location_type == 'dir':
                file_type = 'image_dir'
            elif location_type == 'file':
                file_type = os.path.splitext(location)[1]
            else:
                if location[-1] == '/':
                    raise NonfileURLError()
                file_type = os.path.splittext(location.split('/')[-1])[1]

        self.location = location
        self.location_type = location_type
        self.file_type = file_type
        self.test_size = input_pars.get('test_size', None)
        self.generator = input_pars.get('generator', False)
        if self.generator:
            try:
                self.generator_batch = int(
                    input_pars.get('generator_batch', 1))
            except:
                raise NonIntegerBatchSizeError()
        self.X_cols = input_pars.get('X_cols', None)
        self.y_cols = input_pars.get('y_cols', None)
        self.misc_cols = input_pars.get('misc_cols', None)

    def _load_data(self, loader):
        data_loader = loader.pop('data_loader', None)
        if data_loader is None:
            try:
                data_loader = self.default_loaders[self.file_type]
            except KeyError:
                raise UndeterminableDataLoaderError()
        try:
            loader_function = load_function(data_loader)
            assert (callable(loader_function))
        except:
            raise InvalidDataLoaderFunctionError(data_loader)
        if self.location_type == 'file':
            if self.generator:
                if self.file_type == 'csv':
                    if loader_function == pd.read_csv:
                        loader['chunksize'] = loader.get(
                            'chunksize', self.generator_batch)
            loader_arg = self.location

        if self.location_type == 'url':
            if self.file_type == 'csv' and loader_function == pd.read_csv:
                data = loader_function(self.location, **loader)
            else:
                downloader = Downloader(url)
                downloader.download(out_path)
                filename = self.location.split('/')[-1]
                loader_arg = out_path + '/' + filename

        if self.file_type == 'npz' and loader_function == np.load:
            loader['allow_pickle'] = True
        data = loader_function(loader_arg, **loader)
        if self.location_type == 'directory':
            data = self._image_directory_load(self.location, self.generator)
        if self.file_type == 'npz' and loader_function == np.load:
            data = [data[f] for f in data.files]
        return data

    def _image_directory_load(self, directory, generator):
        # To be overridden by loaders.
        pass

    def _interpret_processor(self, preprocessor, data):
        if preprocessor is None:
            return interpreted_data
        if isinstance(preprocessor, list):
            for p in preprocessor:
                preprocessed_data = self._preprocessor(data, p)
        else:
            preprocessed_data = self._preprocessor(data, p)
        return preprocessed_data

    def _preprocessor(self, data, pars):
        try:
            data_preprocessor = pars.pop('data_preprocessor')
        except KeyError:
            raise MissingDataPreprocessorError()
        if isinstance(data_preprocessor, list):
            preprocessors = []
            output = PreprocssingOutputDict()
            for encoder_pars in data_preprocessor:
                try:
                    index = encoder_pars.pop('index')
                except:
                    raise EncoderMissingIndexError(encoder_pars)
                try:
                    encoder_str = encoder_pars.pop('encoder')
                except:
                    raise EncoderMissingEncoderError(encoder_pars)
                output_name = encoder_pars.pop('output_name', index)
                if isinstance(index, str) or (isinstance(index, list) and
                                              isinstance(index[0], str)):
                    selection = data[index]
                else:
                    try:
                        selection = data[:, index]
                    except:
                        selection = data[index]
                for x, y in encoder_pars.items():
                    parameter_string = str(x) + ' : ' + str(y)
                    if '{data}' in parameter_string:
                        try:
                            encoder_pars[x] = eval(
                                y.replace('{data}', 'selection'))
                        except:
                            raise InvalidEncoderParameterError(
                                parameter_string)
                    elif str(y)[0] == '@':
                        try:
                            encoder_pars[x] = eval(y[1:])
                        except:
                            raise InvalidEncoderParameterError(
                                parameter_string)
                    else:
                        encoder_pars[x] = y
                try:
                    if '{data}' in encoder_str:
                        encoder = lambda x, args: eval(
                                  encoder_str.replace('{data}',
                                                      'selection'))(**args)
                    encoder = load_function(encoder_str)
                except:
                    raise InvalidEncoderError(encoder)
                try:
                    assert callable(encoder)
                except:
                    raise NonCallableEncoderError(encoder_str)
                if inspect.isclass(encoder):
                    encoder_output = encoder.fit_transform(
                        selection, **encoder_pars)
                    preprocessors.append((index, encoder.transform,
                                          output_name, encoder_pars))
                    encoder_output = encoder(selection, **encoder_pars)
                    output[output_name] = encoder_output
                else:
                    preprocessors.append((index, encoder, output_name,
                                          encoder_pars))
                    encoder_output = encoder(selection, **encoder_pars)
                    output[output_name] = encoder_output
            self.state_preprocessors.append(preprocessors)
            return output
        parameters = {}
        for x, y in pars.items():
            parameter_string = str(x) + ' : ' + str(y)
            if '{data}' in parameter_string:
                try:
                    parameters[x] = eval(y.replace('{data}', 'data'))
                except:
                    raise InvalidDataPreprocessorError(parameter_string)
            elif str(y)[0] == '@':
                try:
                    parameters[x] = eval(y[1:])
                except:
                    raise InvalidDataPreprocessorError(parameter_string)
            else:
                parameters[x] = y
        try:
            if '{data}' in data_preprocessor:
                return eval(data_preprocessor.replace('{data}',
                                                      'data'))(**parameters)
            preprocessor = load_function(data_preprocessor)
        except:
            raise InvalidDataPreprocessorError(data_preprocessor)
        try:
            assert callable(preprocessor)
        except:
            raise NonCallableDataPreprocessorError(data_preprocessor)
        if inspect.isclass(preprocessor):
            preprocessor_instance = preprocessor()
            data = preprocessor_instance.fit_transform(data, **parameters)
            self.state_preprocessors.append((preprocessor_instance.transform,
                                             parameters))
            return data
        else:
            data = preprocessor(data, **parameters)
            self.state_preprocessors.append((preprocessor, parameters))
            return data

    def preprocess_new_data(self, data):
        for processor in self.state_preprocessors:
            if inspect.isfunction(processor):
                data = processor(data)
            elif isinstance(processor, tuple):
                data = processor[0](data, **processor[1])
            else:
                output = PreprocssingOutputDict()
                for index, encoder, output_name, args in processor:
                    if isinstance(index,
                                  str) or (isinstance(index, list) and
                                           isinstance(index[0], str)):
                        selection = data[index]
                    else:
                        try:
                            selection = data[:, index]
                        except:
                            selection = data[index]
                    outout[output_name] = encoder(selection, **args)
                data = output
        return output

    def _interpret_output(self, output):
        shape = output.get('shape', None)
        if shape is not None:
            if isinstance(shape[0], list):
                for s, o in zip(shape, self.intermediate_output):
                    if tuple(s) != o.shape:
                        raise OutputShapeError(tuple(s), o.shape)
            elif tuple(shape) != self.intermediate_output.shape:
                raise OutputShapeError(
                    tuple(shape), self.intermediate_output.shape)
        path = output.get('path', None)
        if isinstance(path, str):
            if isinstance(self.intermediate_output, np.ndarray):
                np.save(path, self.intermediate_output)
            elif isinstance(self.intermediate_output, pd.core.frame.DataFrame):
                self.intermediate_output.to_csv(path)
            elif isinstance(self.intermediate_output, list) and \
                all([isinstance(x, np.ndarray)
                    for x in self.intermediate_output]):
                np.savez(path, *self.intermediate_output)
            else:
                pickle.dump(self.intermediate_output, open(path, 'wb'))
        elif isinstance(path, list):
            for p, f in zip(path, self.intermediate_output):
                if isinstance(f, np.ndarray):
                    np.save(p, self.f)
                elif isinstance(f, pd.core.frame.DataFrame):
                    f.to_csv(f)
                elif isinstance(f, list) and all(
                        [isinstance(x, np.ndarray) for x in f]):
                    np.savez(p, *f)
                else:
                    pickle.dump(f, open(p, 'wb'))


class TensorflowDataLoader(AbstractDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create a tf.data.Dataset object


class KerasDataLoader(AbstractDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create a model.fit_generator-compatible generator


class PyTorchDataLoader(AbstractDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create a dynamic Dataset subclass


class GluonTSDataLoader(AbstractDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    print('Dataloader test')
    j = json.loads(
        open('./dataset/json_/namentity_crm_bilstm_dataloader.json').read())[
            'data_pars']
    input_pars = j['input']
    loader = j['loader']
    preprocessing = j['preprocessor']
    output = j['output']
    g = AbstractDataLoader(input_pars, loader, preprocessing, output)
    print(g.intermediate_output)
