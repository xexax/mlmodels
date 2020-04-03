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

    def __init__(self, input_pars, loader, preprocessor, output, **args):
        self._misc_dict = args if args is not None else {}
        self._interpret_input_pars(input_pars)
        loaded_data = self._load_data(loader)
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

        if (
            self.test_size > 0
            and not self.generator
            and len(Xinput) > 0
            and len(yinput) > 0
        ):
            processed_data = (
                train_test_split(*(Xinput + yinput), test_size=self.test_size)
                + misc_input
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
                self.generator_batch = int(input_pars.get("generator_batch", 1))
            except:
                raise NonIntegerBatchSizeError()
        self.X_cols = input_pars.get("X_cols", None)
        self.y_cols = input_pars.get("y_cols", None)
        self.misc_cols = input_pars.get("misc_cols", None)

    def _load_data(self, loader):
        data_loader = loader.pop("data_loader", None)
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
                        loader["chunksize"] = loader.get(
                            "chunksize", self.generator_batch
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

        data = loader_function(loader_arg, **loader)
        if self.path_type == "directory":
            data = self._image_directory_load(self.path, self.generator)
        if self.file_type == "npz" and loader_function == np.load:
            data = [data[f] for f in data.files]
        return data

    def _image_directory_load(self, directory, generator):
        # To be overridden by loaders, or possibly use Keras's ImageDataGenerator by default, as it's designed to be universal
        pass

    def preprocess_new_data(self, data):
        return self.preprocessor.transform(data)

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
                    if hasattr(o,'shape') and tuple(s) != o.shape:
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


###functions for the example JSON test
def test_pandas_fillna(data, **args):
    return data.fillna(**args)


def test_onehot_sentences(data, max_len):
    return (
        lambda df, max_len: (
            lambda d, ml, word_dict, sentence_groups: np.array(
                keras.preprocessing.sequence.pad_sequences(
                    [
                        [word_dict[x] for x in sw]
                        for sw in [y.values for _, y in sentence_groups["Word"]]
                    ],
                    ml,
                    padding="post",
                    value=0,
                    dtype="int",
                ),
                dtype="O",
            )
        )(
            data,
            max_len,
            {y: x for x, y in enumerate(["PAD", "UNK"] + list(data["Word"].unique()))},
            data.groupby("Sentence #"),
        )
    )(data, max_len)


def test_word_count(data):
    return data["Word"].nunique()+2


def test_word_categorical_labels_per_sentence(data, max_len):
    return (
        lambda df, max_len: (
            lambda d, ml, c, tag_dict, sentence_groups: np.array(
                [
                    keras.utils.to_categorical(i, num_classes=c + 1)
                    for i in keras.preprocessing.sequence.pad_sequences(
                        [
                            [tag_dict[w] for w in s]
                            for s in [y.values for _, y in sentence_groups["Tag"]]
                        ],
                        ml,
                        padding="post",
                        value=0,
                    )
                ]
            )
        )(
            data,
            max_len,
            data["Tag"].nunique(),
            {y: x for x, y in enumerate(["PAD"] + list(data["Tag"].unique()))},
            data.groupby("Sentence #"),
        )
    )(data, max_len)


if __name__ == "__main__":
    from models import test_module

    param_pars = {
        "choice": "json",
        "config_mode": "test",
        "data_path": f"dataset/json_/namentity_crm_bilstm_dataloader.json",
    }

    test_module("model_keras/namentity_crm_bilstm_dataloader.py", param_pars)
