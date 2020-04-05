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
import cloudpickle as pickle
from preprocessor import Preprocessor
from util import load_callable_from_dict
import re
import spacy
import torch
import torchtext
from torchtext.data import Field, Pipeline
from mlmodels.util import log
from time import sleep


class DataLoaderError(Exception):
    pass


class MissingLocationKeyError(DataLoaderError):
    def __init__(self):
        print("Location key missing from the input dictionary.")


class UndeterminableLocationTypeError(DataLoaderError):
    def __init__(self, location):
        print(f"Location type cannot be inferred for '{location}'.")


class UnknownLocationTypeError(DataLoaderError):
    def __init__(self, path_type):
        print(f"Location type '{  path_type}' is unknown.")


class NonfileURLError(DataLoaderError):
    def __init__(self):
        print(f"URL must point to a file.")


class UndeterminableDataLoaderError(DataLoaderError):
    def __init__(self):
        print(
            f"""Loader function to be used was not provided and could not be
             automatically inferred from file type."""
        )


class NonIntegerBatchSizeError(DataLoaderError):
    def __init__(self):
        print(f"Provided batch size cannot be interpreted as an integer.")


class InvalidDataLoaderFunctionError(DataLoaderError):
    def __init__(self, loader):
        print(f"Invalid data loader function '{loader}\ specified.")


class NumpyGeneratorError(DataLoaderError):
    def __init__(self):
        print(f"Loading Numpy binaries as generators is unsupported.")


class OutputShapeError(DataLoaderError):
    def __init__(self, specified, actual):
        print(
            f"""Specified output shape {specified} does not match actual output
            shape {actual}"""
        )


def open_read(file):
    return open(f).read()


def pickle_load(file):
    return pickle.load(open(f, " r"))


class AbstractDataLoader:

    default_loaders = {
        ".csv": {"uri": "pandas::read_csv"},
        ".txt": {"uri": "dataloader::open_read"},
        ".npy": {"uri": "numpy::load"},
        ".npz": {"uri": "np::.load", "arg": {"allow_pickle": True}},
        ".pkl": {"uri": "dataloader::pickle_load"},
    }

    validation_split_function = [train_test_split, "test_size", {}]

    def __init__(self, input_pars, loader, preprocessor, output, **args):
        self._misc_dict = args if args is not None else {}
        self._interpret_input_pars(input_pars)
        loaded_data = self._load_data(loader)
        if isinstance(preprocessor, Preprocessor):
            self.preprocessor = preprocessor
            processed_data = self.preprocessor.transform(loaded_data)

        else:
            self.preprocessor = Preprocessor(preprocessor)
            processed_data = self.preprocessor.fit_transform(loaded_data)

        ######## Xinput
        Xinput = []
        if self.X_cols is not None:
            for x in self.X_cols:
                if isinstance(x, str):
                    Xinput.append(processed_data[x])
                else:
                    try:
                        Xinput.append(processed_data[:, x])
                    except:
                        Xinput.append(processed_data[x])

        ######## yinput
        yinput = []
        if self.y_cols is not None:
            for y in self.y_cols:
                if isinstance(y, str):
                    yinput.append(processed_data[y])
                else:
                    try:
                        yinput.append(processed_data[:, y])
                    except:
                        yinput.append(processed_data[y])

        ####### Other columns (columns that shouldn't be split if test_size > 0)
        misc_input = []
        if self.misc_cols is not None:
            for m in self.misc_cols:
                if isinstance(m, str):
                    misc_input.append(processed_data[m])
                else:
                    try:
                        misc_input.append(processed_data[:, m])
                    except:
                        misc_input.append(processed_data[m])

        split_size_arg_dict = {
            self.validation_split_function[1]: self.test_size,
            **self.validation_split_function[2],
        }
        if (
            self.test_size > 0
            and not self.generator
            and len(Xinput) > 0
            and len(yinput) > 0
        ):

            processed_data = self.validation_split_function[0](
                *(Xinput + yinput), **split_size_arg_dict
            )
        elif self.test_size > 0:
            processed_data = self.validation_split_function[0](
                processed_data, **split_size_arg_dict
            )

        if isinstance(processed_data, Preprocessor.output_dict_type):
            processed_data = list(processed_data.values())

        if isinstance(processed_data, list) and len(processed_data) == 1:
            processed_data = processed_data[0]

        self.intermediate_output = processed_data
        self._interpret_output(output)

    def __getitem__(self, key):
        return self._misc_dict[key]

    def _interpret_input_pars(self, input_pars):
        try:
            path = input_pars["path"]
        except KeyError:
            raise MissingLocationKeyError()

        path_type = input_pars.get("path_type", None)
        if path_type is None:
            if os.path.isfile(path):
                path_type = "file"
            if os.path.isdir(path):
                path_type = "dir"
            if urlparse(path).scheme != "":
                path_type = "url"
                download_path = input_pars.get("download_path", "./")
            if path_type == "dropbox":
                dropbox_download(path)
                path_type = "file"
            if path_type is None:
                raise UndeterminableLocationTypeError(path)

        elif path_type != "file" and path_type != "dir" and path_type != "url":
            raise UnknownLocationTypeError()

        file_type = input_pars.get("file_type", None)
        if file_type is None:
            if path_type == "dir":
                file_type = "image_dir"
            elif path_type == "file":
                file_type = os.path.splitext(path)[1]
            else:
                if path[-1] == "/":
                    raise NonfileURLError()
                file_type = os.path.splittext(path.split("/")[-1])[1]

        self.path = path
        self.path_type = path_type
        self.file_type = file_type
        self.test_size = input_pars.get("test_size", None)
        self.generator = input_pars.get("generator", False)
        if self.generator:
            try:
                self.batch_size = int(input_pars.get("batch_size", 1))
            except:
                raise NonIntegerBatchSizeError()
        self.X_cols = input_pars.get("X_cols", None)
        self.y_cols = input_pars.get("y_cols", None)
        self.misc_cols = input_pars.get("misc_cols", None)

    def _load_data(self, loader):
        data_loader = loader.get("data_loader", None)
        if isinstance(data_loader, tuple):
            loader_function = data_loader[0]
            loader_args = data_loader[1]
        else:
            if data_loader is None:
                try:
                    data_loader = self.default_loaders[self.file_type]
                except KeyError:
                    raise UndeterminableDataLoaderError()
            try:
                loader_function, loader_args = load_callable_from_dict(data_loader)
                assert callable(loader_function)
            except:
                raise InvalidDataLoaderFunctionError(data_loader)
        if self.path_type == "file":
            if self.generator:
                if self.file_type == "csv":
                    if loader_function == pd.read_csv:
                        loader_args["chunksize"] = loader.get(
                            "chunksize", self.batch_size
                        )
            loader_arg = self.path

        if self.path_type == "url":
            if self.file_type == "csv" and loader_function == pd.read_csv:
                data = loader_function(self.path, **loader_args)
            else:
                downloader = Downloader(url)
                downloader.download(out_path)
                filename = self.path.split("/")[-1]
                loader_arg = out_path + "/" + filename

        data = loader_function(loader_arg, **loader_args)
        if self.path_type == "directory":
            data = self._image_directory_load(self.path, self.generator)
        if self.file_type == "npz" and loader_function == np.load:
            data = [data[f] for f in data.files]
        return data

    def _image_directory_load(self, directory, generator):
        # To be overridden by loaders, or possibly use Keras's ImageDataGenerator by default, as it's designed to be universal
        pass

    def _interpret_output(self, output):
        max_len = output.get("out_max_len", None)
        intermediate_output = self.intermediate_output
        if max_len is not None:
            try:
                intermediate_output = self.intermediate_output[0:max_len]
            except:
                pass
        shape = output.get("shape", None)
        if shape is not None:
            if isinstance(shape[0], list):
                for s, o in zip(shape, intermediate_output):
                    if hasattr(o, "shape") and tuple(s) != o.shape:
                        raise OutputShapeError(tuple(s), o.shape)
            elif tuple(shape) != intermediate_output.shape:
                raise OutputShapeError(tuple(shape), intermediate_output.shape)
            self.output_shape = shape
        path = output.get("path", None)

        if isinstance(path, str):
            if isinstance(intermediate_output, np.ndarray):
                np.save(path, intermediate_output)
            elif isinstance(intermediate_output, pd.core.frame.DataFrame):
                intermediate_output.to_csv(path)
            elif isinstance(intermediate_output, list) and all(
                [isinstance(x, np.ndarray) for x in intermediate_output]
            ):
                np.savez(path, *intermediate_output)
            else:
                pickle.dump(intermediate_output, open(path, "wb"))

        elif isinstance(path, list):
            for p, f in zip(path, intermediate_output):
                if isinstance(f, np.ndarray):
                    np.save(p, self.f)
                elif isinstance(f, pd.core.frame.DataFrame):
                    f.to_csv(f)
                elif isinstance(f, list) and all(
                    [isinstance(x, np.ndarray) for x in f]
                ):
                    np.savez(p, *f)
                else:
                    pickle.dump(f, open(p, "wb"))

    def get_data(self):
        return self.intermediate_output


class TensorflowDataLoader(AbstractDataLoader):
    def __init__(self, *args, **kwargs):
        super(TensorflowDataLoader, self).__init__(*args, **kwargs)
        # Create a tf.data.Dataset object


class KerasDataLoader(AbstractDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create a model.fit_generator-compatible generator


class PyTorchDataLoader(AbstractDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.generator:
            if isinstance(self.intermediate_output, list) or isinstance(
                self.intermediate_output, tuple
            ):
                self.intermediate_output = [
                    torchtext.data.Iterator(
                        output,
                        self.batch_size,
                        device=torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        ),
                    )
                    for output in self.intermediate_output
                ]
            else:
                if isinstance(
                    self.intermediate_output, torchtext.data.Dataset
                ) or isinstance(
                    self.intermediate_output, torchtext.data.TabularDataset
                ):
                    self.intermediate_output = torchtext.data.Iterator(
                        self.intermediate_outputs,
                        self.batch_size,
                        device=torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        ),
                    )

    def _load_data(self, loader):
        # if loader is tabulardataset then assume its use. otherwise, assume that the loader's output is to be fed to dataset. change the split function.
        if "torchtext_fields" in loader.keys():
            self.validation_split_function = [
                lambda x, test_size, **args: x.split(test_size, **args),
                "test_size",
                {},
            ]
            fields = []
            field_vocab_args = {}
            for field_pars in loader["torchtext_fields"]:
                # process each field
                field_types = {"Field": Field}
                field_name = field_pars["name"]
                field_class = field_types[field_pars.get("type", "Field")]
                field_args = field_pars.get("args", {})
                if "preprocessing" in field_pars.keys():
                    processor, _ = load_callable_from_dict(field_pars["preprocessing"])
                    field_args["preprocessing"] = Pipeline(processor)
                if "postprocessing" in field_pars.keys():
                    processor, _ = load_callable_from_dict(field_pars["postprocessing"])
                    field_args["postprocessing"] = Pipeline(processor)
                if "tokenize" in field_pars.keys():
                    tokenizer, _ = load_callable_from_dict(field_pars["tokenize"])
                    field_args["tokenize"] = tokenizer
                field = field_class(**field_args)
                fields.append((field_name, field))
                field_vocab_args[field_name] = field_pars.get("vocab_args", {})
            loader_function = None
            if "data_loader" in loader.keys():
                data_loader = loader["data_loader"]
                loader_function, loader_args = load_callable_from_dict(data_loader)
                if loader_function != torchtext.data.TabularDataset:
                    data = torchtext.data.Dataset(
                        super(PyTorchDataLoader, self)._load_data(loader),
                        fields,
                        **loader_args,
                    )
                else:
                    loader_args["fields"] = fields
                    loader_args["format"] = "csv"
                    loader_args["skip_header"] = True
                    loader = loader.copy()
                    loader["data_loader"] = (torchtext.data.TabularDataset, arg_dict)
                    data = super(PyTorchDataLoader, self)._load_data(loader)
            else:
                loader = loader.copy()
                loader["data_loader"] = (
                    torchtext.data.TabularDataset,
                    {"fields": fields, "format": "csv", "skip_header": True},
                )
                data = super(PyTorchDataLoader, self)._load_data(loader)
            for f in fields:
                f[1].build_vocab(data, **field_vocab_args[f[0]])
            return data
        else:
            return super(PyTorchDataLoader, self)._load_data(self, loader)

    def _interpret_output(self, output):
        # if the intermediate data is a torch Dataset or Iterator, handle special logic. otherwise, default.
        super(PyTorchDataLoader, self)._interpret_output(output)


class GluonTSDataLoader(AbstractDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


###functions for the example JSON tests

# namentity_crm_bilstm_dataloader



if __name__ == "__main__":
    from models import test_module

    # param_pars = {
    #    "choice": "json",
    #    "config_mode": "test",
    #    "data_path": f"dataset/json_/namentity_crm_bilstm_dataloader.json",
    # }

    # test_module("model_keras/namentity_crm_bilstm_dataloader.py", param_pars)

    param_pars = {
        "choice": "json",
        "config_mode": "test",
        "data_path": f"dataset/json_/textcnn_dataloader.json",
    }
    test_module("model_tch/textcnn_dataloader.py", param_pars)
