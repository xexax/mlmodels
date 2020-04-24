""""

Related to data procesisng



"""
import os, Path
import pandas as pd, numpy as np


from mlmodels.util import path_norm



###############################################################################################################
###############################################################################################################
def torch_datasets_wrapper(sets, args_list = None, **args):
    if not isinstance(sets,list) and not isinstance(sets,tuple):
        sets = [sets]
    import torch
    if args_list is None:
        return [torch.utils.data.DataLoader(x,**args) for x in sets]
    return [torch.utils.data.DataLoader(x,**a,**args) for a,x in zip(args_list,sets)]



def load_function(package="mlmodels.util", name="path_norm"):
  import importlib
  return  getattr(importlib.import_module(package), name)


def get_dataset_torch(data_pars):
    """"
      torchvison.datasets
         MNIST Fashion-MNIST KMNIST EMNIST QMNIST  FakeData COCO Captions Detection LSUN ImageFolder DatasetFolder 
         ImageNet CIFAR STL10 SVHN PhotoTour SBU Flickr VOC Cityscapes SBD USPS Kinetics-400 HMDB51 UCF101 CelebA

      torchtext.datasets
         Sentiment Analysis
         SST IMDb Question Classification TREC Entailment SNLI MultiNLI 
         Language Modeling WikiText-2 WikiText103 
         PennTreebank 
         Machine Translation 
            Multi30k IWSLT WMT14 
         Sequence Tagging 
            UDPOS CoNLL2000Chunking 
         Question Answering 
            BABI20


       from  mlmodels.preprocess.image import pandas_dataset

       dset = pandas_dataset(data_pars)


    """
    import torch
    d = data_pars

    transform = None    
    if  d["transform"]  :
       transform = load_function(  d.get("preprocess_module", "mlmodels.preprocess.image"), 
                                   d.get("transform", "torch_transform_mnist" ))()


    ##### Dataset from file available
    train_loader, valid_loader = None, None
    if d['train_path'] and  d['test_path'] :
      # Load from files  
      """
        Tabular Dataset from numpy files or from CSV files.
        Text,Tabular or Image


      """
      myDataset_fromfiles = None
      dset = myDataset_fromfiles(data_pars)
      train_loader = torch.utils.data.DataLoader( dset(d['data_path'], train=True, download=True, transform= transform),
                                                  batch_size=d['train_batch_size'], shuffle=True)
      # Load from files  
      myDataset_fromfiles = None
      dset = myDataset_fromfiles(data_pars)
      valid_loader = torch.utils.data.DataLoader( dset(d['data_path'], train=False, download=True, transform= transform),
                                                  batch_size=d['train_batch_size'], shuffle=True)

      ### To Finish 




    else :
      raise Exception("Issues with train_path, test_path")


    ###### Pre Build Dataset available 
    if d['dataset'] :
        dataset_module =  d.get('dataset_module', "torchvision.datasets")   
        dset = load_function(dataset_module, d["dataset"] )

        train_loader = torch.utils.data.DataLoader( dset(d['data_path'], train=True, download=True, transform= transform),
                                                    batch_size=d['train_batch_size'], shuffle=True)
        
        valid_loader = torch.utils.data.DataLoader( dset(d['data_path'], train=False, download=True, transform= transform),
                                                    batch_size=d['train_batch_size'], shuffle=True)


    return train_loader, valid_loader  









class pandasDataset(Dataset):
    """
    Defines a dataset composed of sentiment text and labels
    Attributes:
        df (Dataframe): Dataframe of the CSV from teh path
        vocab (dict{str: int}: A vocabulary dictionary from word to indices for this dataset
        sample_weights(ndarray, shape(len(labels),)): An array with each sample_weight[i] as the weight of the ith sample
        data (list[int, [int]]): The data in the set
    """
   
    def __init__(self, data_pars):
        import torch
        self.data_pars = data_pars
        d = data_pars
        path = d['path']
        df = pd.read_csv(path, header=None, names=['stars', 'text'])

        self.df = df

        process_custom = load_function( "mlmodels.preprocess.generic",    )()
        

        # Split
        X = df[ data_pars["colX"] ]
        labels = df[ data_pars["coly"] ]

        
        # compute sample weights from inverse class frequencies
        class_sample_count = np.unique(labels, return_counts=True)[1]
        weight = 1. / class_sample_count
        self.samples_weight = torch.from_numpy(weight[labels])


        #### Data
        self.data = list(zip(labels, X, df["lengths"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]




def get_loader(fix_length, vocab_threshold, batch_size):
    train_dataset = SentimentDataset("data/train.csv", fix_length, vocab_threshold)

    vocab = train_dataset.vocab

    # valid_dataset = SentimentDataset("data/valid.csv", fix_length, vocab_threshold, vocab)

    test_dataset = SentimentDataset("data/test.csv", fix_length, vocab_threshold, vocab)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4)

    """
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4)
    """

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4)

    return train_dataloader, test_dataloader, vocab





#########################################################################################################
#########################################################################################################
class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    from torchvision.datasets.vision import VisionDataset
    import warnings
    from PIL import Image
    import os
    import os.path
    import numpy as np
    import torch
    import codecs
    import string
    from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive,  verify_str_arg


    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")






#########################################################################################################
#########################################################################################################
def pandas_dataset() :

    from typing import Union, Dict

    import pandas as pd
    from torchtext.data import (Field, Example, Iterator, BucketIterator, Dataset)
    from tqdm import tqdm


    class DataFrameExampleSet(Dataset):
        def __init__(self, data_pars):
            self.data_pars = data_pars
            d =data_pars
            self._df = pd.read_csv( d['load_path']  )

            fields = None
            self._fields = fields
            self._fields_dict = {field_name: (field_name, field)
                                 for field_name, field in fields.items()
                                 if field is not None}

        def __iter__(self):
            for item in tqdm(self._df.itertuples(), total=len(self)):
                example = Example.fromdict(item._asdict(), fields=self._fields_dict)
                yield example

        def __len__(self):
            return len(self._df)

        def shuffle(self, random_state=None):
            self._df = self._df.sample(frac=1.0, random_state=random_state)


    class DataFrameDataset(Dataset):
        def __init__(self, df: pd.DataFrame,
                     fields: Dict[str, Field], filter_pred=None):
            examples = DataFrameExampleSet(df, fields)
            super().__init__(examples, fields, filter_pred=filter_pred)


    class DataFrameIterator(Iterator):
        def data(self):
            if isinstance(self.dataset.examples, DataFrameExampleSet):
                if self.shuffle:
                    self.dataset.examples.shuffle()
                dataset = self.dataset
            else:
                dataset = super().data()
            return dataset


    class DataFrameBucketIterator(BucketIterator):
        def data(self):
            if isinstance(self.dataset.examples, DataFrameExampleSet):
                if self.shuffle:
                    self.dataset.examples.shuffle()
                dataset = self.dataset
            else:
                dataset = super().data()
            return dataset




def custom_dataset():
    import re
    import logging

    import numpy as np
    import pandas as pd
    import spacy
    import torch
    from torchtext import data

    NLP = spacy.load('en')
    MAX_CHARS = 20000
    VAL_RATIO = 0.2
    LOGGER = logging.getLogger("toxic_dataset")



    def get_dataset(fix_length=100, lower=False, vectors=None):
        if vectors is not None:
            # pretrain vectors only supports all lower cases
            lower = True


        comment = data.Field(
            sequential=True,
            fix_length=fix_length,
            tokenize=tokenizer,
            pad_first=True,
            tensor_type=torch.cuda.LongTensor,
            lower=lower
        )

        train, val = data.TabularDataset.splits(
            path='cache/', format='csv', skip_header=True,
            train='dataset_train.csv', validation='dataset_val.csv',
            fields=[
                ('id', None),
                ('comment_text', comment),
                ('toxic', data.Field(
                    use_vocab=False, sequential=False, tensor_type=torch.cuda.ByteTensor)),
                ('severe_toxic', data.Field(
                    use_vocab=False, sequential=False, tensor_type=torch.cuda.ByteTensor)),
                ('obscene', data.Field(
                    use_vocab=False, sequential=False, tensor_type=torch.cuda.ByteTensor)),
                ('threat', data.Field(
                    use_vocab=False, sequential=False, tensor_type=torch.cuda.ByteTensor)),
                ('insult', data.Field(
                    use_vocab=False, sequential=False, tensor_type=torch.cuda.ByteTensor)),
                ('identity_hate', data.Field(
                    use_vocab=False, sequential=False, tensor_type=torch.cuda.ByteTensor)),
            ])

        test = data.TabularDataset(
            path='cache/dataset_test.csv', format='csv', skip_header=True,
            fields=[
                ('id', None),
                ('comment_text', comment)
            ])
        LOGGER.debug("Building vocabulary...")
        comment.build_vocab(
            train, val, test,
            max_size=20000,
            min_freq=50,
            vectors=vectors
        )
        return train, val, test


    def get_iterator(dataset, batch_size, train=True, shuffle=True, repeat=False):
        dataset_iter = data.Iterator(
            dataset, batch_size=batch_size, device=0,
            train=train, shuffle=shuffle, repeat=repeat,
            sort=False
        )
        return dataset_iter




def text_dataloader():  
    import torchtext
    import torchtext.data as data
    import torchtext.vocab as vocab
    import os
    import spacy
    import pandas as pd
    import random
    import dill
    from tqdm import tqdm
    from torchtext.data import BucketIterator

    spacy = spacy.load("en_core_web_sm")
    SEED = 1024

    def spacy_tokenize(x):
        return [
            tok.text
            for tok in spacy.tokenizer(x)
            if not tok.is_punct | tok.is_space
        ]


    class NewsDataset(data.Dataset):
        def __init__(
            self, path, max_src_len=100, field=None, debug=False, **kwargs
        ):
            examples = []
            fields = [("src", field), ("tgt", field)]
            df = pd.read_csv(path, encoding="utf-8", usecols=["content", "title"])
            df = df[~(df["content"].isnull() | df["title"].isnull())]
            df = df[~(df["content"] == "[]")]
            for i in tqdm(range(df.shape[0])):
                examples.append(
                    data.Example.fromlist(
                        [df.iloc[i].content, df.iloc[i].title], fields
                    )
                )
                if debug and i == 100:
                    break
            super().__init__(
                examples, fields, filter_pred=lambda s: len(s.src) > 10, **kwargs
            )
            for example in self.examples:
                example.tgt = ["<sos>"] + example.tgt + ["<eos>"]


    class NewsDataLoader:
        def __init__(
            self,
            csv_path,
            use_save=False,
            embed_path=None,
            build_vocab=True,
            batch_size=64,
            val_size=0.2,
            max_src_len=100,
            save=True,
            shuffle=True,
            debug=False,
        ):
            random.seed(SEED)

            def trim_sentence(s):
                return s[:max_src_len]

            self.field = data.Field(
                tokenize=spacy_tokenize,
                batch_first=True,
                include_lengths=True,
                lower=True,
                preprocessing=trim_sentence,
            )

            if use_save:
                save = False
                build_vocab = False
                with open("data/dataset.pickle", "rb") as f:
                    self.field = dill.load(f)

            dataset = NewsDataset(
                csv_path, max_src_len, field=self.field, debug=debug
            )

            if build_vocab:
                # load custom word vectors
                if embed_path:
                    path, embed = os.path.split(embed_path)
                    vec = vocab.Vectors(embed, cache=path)
                    self.field.build_vocab(dataset, vectors=vec)
                else:
                    self.field.build_vocab(
                        dataset, vectors="glove.6B.300d", max_size=40000
                    )

            self.dataloader = BucketIterator(
                dataset,
                batch_size=batch_size,
                device=-1,
                sort_key=lambda x: len(x.src),
                sort_within_batch=True,
                repeat=False,
                shuffle=shuffle,
            )
            self.stoi = self.field.vocab.stoi
            self.itos = self.field.vocab.itos
            self.sos_id = self.stoi["<sos>"]
            self.eos_id = self.stoi["<eos>"]
            self.pad_id = self.stoi["<pad>"]
            self.n_examples = len(dataset)

            if save:
                with open("data/dataset.pickle", "wb") as f:
                    temp = self.field
                    dill.dump(temp, f)

        def __iter__(self):
            for batch in self.dataloader:
                x = batch.src
                y = batch.tgt
                yield (x[0], x[1], y[0], y[1])

        def __len__(self):
            return len(self.dataloader)


    if __name__ == "__main__":
        dl = NewsDataLoader(csv_path="data/val.csv", debug=True, save=False)
        for x, len_x, y, len_y in dl:
            if (len_x == 0).any():
                import pdb

                pdb.set_trace()






###############################################################################################################
def tf_dataset(dataset_pars):
    """
        Save in numpy compressez format TF Datasets
    
        dataset_pars ={ "dataset_id" : "mnist", "batch_size" : 5000, "n_train": 500, "n_test": 500, 
                            "out_path" : "dataset/vision/mnist2/" }
        tf_dataset(dataset_pars)
        
        
        https://www.tensorflow.org/datasets/api_docs/python/tfds
        import tensorflow_datasets as tfds
        import tensorflow as tf
        
        # Here we assume Eager mode is enabled (TF2), but tfds also works in Graph mode.
        print(tfds.list_builders())
        
        # Construct a tf.data.Dataset
        ds_train = tfds.load(name="mnist", split="train", shuffle_files=True)
        
        # Build your input pipeline
        ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
        for features in ds_train.take(1):
          image, label = features["image"], features["label"]
          
          
        NumPy Usage with tfds.as_numpy
        train_ds = tfds.load("mnist", split="train")
        train_ds = train_ds.shuffle(1024).batch(128).repeat(5).prefetch(10)
        
        for example in tfds.as_numpy(train_ds):
          numpy_images, numpy_labels = example["image"], example["label"]
        You can also use tfds.as_numpy in conjunction with batch_size=-1 to get the full dataset in NumPy arrays from the returned tf.Tensor object:
        
        train_ds = tfds.load("mnist", split=tfds.Split.TRAIN, batch_size=-1)
        numpy_ds = tfds.as_numpy(train_ds)
        numpy_images, numpy_labels = numpy_ds["image"], numpy_ds["label"]
        
        
        FeaturesDict({
    'identity_attack': tf.float32,
    'insult': tf.float32,
    'obscene': tf.float32,
    'severe_toxicity': tf.float32,
    'sexual_explicit': tf.float32,
    'text': Text(shape=(), dtype=tf.string),
    'threat': tf.float32,
    'toxicity': tf.float32,
})
            
            
    
    """
    import tensorflow_datasets as tfds
    import numpy as np

    d          = dataset_pars
    dataset_id = d['dataset_id']
    batch_size = d.get('batch_size', -1)  # -1 neans all the dataset
    n_train    = d.get("n_train", 500)
    n_test     = d.get("n_test", 500)
    out_path   = path_norm(d['out_path'] )
    name       = dataset_id.replace(".","-")    
    os.makedirs(out_path, exist_ok=True) 


    train_ds = tfds.as_numpy( tfds.load(dataset_id, split= f"train[0:{n_train}]", batch_size=batch_size) )
    test_ds  = tfds.as_numpy( tfds.load(dataset_id, split= f"test[0:{n_test}]", batch_size=batch_size) )
    # val_ds  = tfds.as_numpy( tfds.load(dataset_id, split= f"test[0:{n_test}]", batch_size=batch_size) )

    print("train", train_ds.shape )
    print("test",  test_ds.shape )

    def get_keys(x):
       if "image" in x.keys() : xkey = "image"
       if "text" in x.keys() : xkey = "text"    
       return xkey
    
    
    for x in train_ds:
       #print(x)
       xkey =  get_keys(x)
       np.savez_compressed(out_path + f"{name}_train" , X = x[xkey] , y = x.get('label') )
        

    for x in test_ds:
       #print(x)
       np.savez_compressed(out_path + f"{name}_test", X = x[xkey] , y = x.get('label') )
        
    print(out_path, os.listdir( out_path ))
        
      





