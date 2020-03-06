import csv
import inspect
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm, trange
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.modeling_gpt2 import GPT2LMHeadModel

from nltk.tokenize.treebank import TreebankWordDetokenizer
from pplm.run_pplm import run_pplm_example
from pplm_classification_head import ClassificationHead
from torchtext import data as torchtext_data
from torchtext import datasets

torch.manual_seed(0)
np.random.seed(0)
EPSILON = 1e-10
example_sentence = "This is incredible! I love it, this is the best chicken I have ever had."
max_length_seq = 100




class Model(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
            self,
            class_size,
            pretrained_model="gpt2-medium",
            cached_mode=False,
            device='cpu'
    ):
        super(Discriminator, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.encoder = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.embed_size = self.encoder.transformer.config.hidden_size
        self.classifier_head = ClassificationHead(
            class_size=class_size,
            embed_size=self.embed_size
        )
        self.cached_mode = cached_mode
        self.device = device

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x):
        mask = x.ne(0).unsqueeze(2).repeat(
            1, 1, self.embed_size
        ).float().to(self.device).detach()
        hidden, _ = self.encoder.transformer(x)
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (
                torch.sum(mask, dim=1).detach() + EPSILON
        )
        return avg_hidden

    def forward(self, x):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
        else:
            avg_hidden = self.avg_representation(x.to(self.device))

        logits = self.classifier_head(avg_hidden)
        probs = F.log_softmax(logits, dim=-1)

        return probs


class Dataset(data.Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        return data


def collate_fn(data):
    def pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_sequences = torch.zeros(
            len(sequences),
            max(lengths)
        ).long()  # padding value = 0

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_sequences[i, :end] = seq[:end]

        return padded_sequences, lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch, _ = pad_sequences(item_info["X"])
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def cached_collate_fn(data):
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch = torch.cat(item_info["X"], 0)
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def fit(data_loader, discriminator, optimizer,
                epoch=0, log_interval=10, device='cpu'):
    samples_so_far = 0
    discriminator.train_custom()
    for batch_idx, (input_t, target_t) in enumerate(data_loader):
        input_t, target_t = input_t.to(device), target_t.to(device)

        optimizer.zero_grad()

        output_t = discriminator(input_t)
        loss = F.nll_loss(output_t, target_t)
        loss.backward(retain_graph=True)
        optimizer.step()

        samples_so_far += len(input_t)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1,
                    samples_so_far, len(data_loader.dataset),
                    100 * samples_so_far / len(data_loader.dataset), loss.item()
                )
            )


def metrics(data_loader, discriminator, device='cpu'):
    discriminator.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for input_t, target_t in data_loader:
            input_t, target_t = input_t.to(device), target_t.to(device)
            output_t = discriminator(input_t)
            # sum up batch loss
            test_loss += F.nll_loss(output_t, target_t, reduction="sum").item()
            # get the index of the max log-probability
            pred_t = output_t.argmax(dim=1, keepdim=True)
            correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()

    test_loss /= len(data_loader.dataset)

    print(
        "Performance on test set: "
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)
        )
    )


def predict(input_sentence, model, classes, cached=False, device='cpu'):
    input_t = model.tokenizer.encode(input_sentence)
    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    if cached:
        input_t = model.avg_representation(input_t)

    log_probs = model(input_t).data.cpu().numpy().flatten().tolist()
    print("Input sentence:", input_sentence)
    print("Predictions:", ", ".join(
        "{}: {:.4f}".format(c, math.exp(log_prob)) for c, log_prob in
        zip(classes, log_probs)
    ))


def get_cached_data_loader(dataset, batch_size, discriminator,
                           shuffle=False, device='cpu'):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)

    xs = []
    ys = []
    for batch_idx, (x, y) in enumerate(tqdm(data_loader, ascii=True)):
        with torch.no_grad():
            x = x.to(device)
            avg_rep = discriminator.avg_representation(x).cpu().detach()
            avg_rep_list = torch.unbind(avg_rep.unsqueeze(1))
            xs += avg_rep_list
            ys += y.cpu().numpy().tolist()

    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(xs, ys),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=cached_collate_fn)

    return data_loader


def test_case_1(
        dataset, dataset_fp=None, pretrained_model="gpt2-medium",
        epochs=10, batch_size=64, log_interval=10,
        save_model=False, cached=False, no_cuda=False):
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    print("Preprocessing {} dataset...".format(dataset))
    start = time.time()

    if dataset == "SST":
        idx2class = ["positive", "negative", "very positive", "very negative",
                     "neutral"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class),
            pretrained_model=pretrained_model,
            cached_mode=cached,
            device=device
        ).to(device)

        text = torchtext_data.Field()
        label = torchtext_data.Field(sequential=False)
        train_data, val_data, test_data = datasets.SST.splits(
            text,
            label,
            fine_grained=True,
            train_subtrees=True,
        )

        x = []
        y = []
        for i in trange(len(train_data), ascii=True):
            seq = TreebankWordDetokenizer().detokenize(
                vars(train_data[i])["text"]
            )
            seq = discriminator.tokenizer.encode(seq)
            seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)
            x.append(seq)
            y.append(class2idx[vars(train_data[i])["label"]])
        train_dataset = Dataset(x, y)

        test_x = []
        test_y = []
        for i in trange(len(test_data), ascii=True):
            seq = TreebankWordDetokenizer().detokenize(
                vars(test_data[i])["text"]
            )
            seq = discriminator.tokenizer.encode(seq)
            seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)
            test_x.append(seq)
            test_y.append(class2idx[vars(test_data[i])["label"]])
        test_dataset = Dataset(test_x, test_y)

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 2,
        }

    elif dataset == "clickbait":
        idx2class = ["non_clickbait", "clickbait"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class),
            pretrained_model=pretrained_model,
            cached_mode=cached,
            device=device
        ).to(device)

        with open("datasets/clickbait/clickbait_train_prefix.txt") as f:
            data = []
            for i, line in enumerate(f):
                try:
                    data.append(eval(line))
                except:
                    print("Error evaluating line {}: {}".format(
                        i, line
                    ))
                    continue
        x = []
        y = []
        with open("datasets/clickbait/clickbait_train_prefix.txt") as f:
            for i, line in enumerate(tqdm(f, ascii=True)):
                try:
                    d = eval(line)
                    seq = discriminator.tokenizer.encode(d["text"])

                    if len(seq) < max_length_seq:
                        seq = torch.tensor(
                            [50256] + seq, device=device, dtype=torch.long
                        )
                    else:
                        print("Line {} is longer than maximum length {}".format(
                            i, max_length_seq
                        ))
                        continue
                    x.append(seq)
                    y.append(d["label"])
                except:
                    print("Error evaluating / tokenizing"
                          " line {}, skipping it".format(i))
                    pass

        full_dataset = Dataset(x, y)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 1,
        }

    elif dataset == "toxic":
        idx2class = ["non_toxic", "toxic"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class),
            pretrained_model=pretrained_model,
            cached_mode=cached,
            device=device
        ).to(device)

        x = []
        y = []
        with open("datasets/toxic/toxic_train.txt") as f:
            for i, line in enumerate(tqdm(f, ascii=True)):
                try:
                    d = eval(line)
                    seq = discriminator.tokenizer.encode(d["text"])

                    if len(seq) < max_length_seq:
                        seq = torch.tensor(
                            [50256] + seq, device=device, dtype=torch.long
                        )
                    else:
                        print("Line {} is longer than maximum length {}".format(
                            i, max_length_seq
                        ))
                        continue
                    x.append(seq)
                    y.append(int(np.sum(d["label"]) > 0))
                except:
                    print("Error evaluating / tokenizing"
                          " line {}, skipping it".format(i))
                    pass

        full_dataset = Dataset(x, y)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 0,
        }

    else:  # if dataset == "generic":
        # This assumes the input dataset is a TSV with the following structure:
        # class \t text

        if dataset_fp is None:
            raise ValueError("When generic dataset is selected, "
                             "dataset_fp needs to be specified aswell.")

        classes = set()
        with open(dataset_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            for row in tqdm(csv_reader, ascii=True):
                if row:
                    classes.add(row[0])

        idx2class = sorted(classes)
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class),
            pretrained_model=pretrained_model,
            cached_mode=cached,
            device=device
        ).to(device)

        x = []
        y = []
        with open(dataset_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(tqdm(csv_reader, ascii=True)):
                if row:
                    label = row[0]
                    text = row[1]

                    try:
                        seq = discriminator.tokenizer.encode(text)
                        if (len(seq) < max_length_seq):
                            seq = torch.tensor(
                                [50256] + seq,
                                device=device,
                                dtype=torch.long
                            )

                        else:
                            print(
                                "Line {} is longer than maximum length {}".format(
                                    i, max_length_seq
                                ))
                            continue

                        x.append(seq)
                        y.append(class2idx[label])

                    except:
                        print("Error tokenizing line {}, skipping it".format(i))
                        pass

        full_dataset = Dataset(x, y)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, test_size]
        )

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 0,
        }

    end = time.time()
    print("Preprocessed {} data points".format(
        len(train_dataset) + len(test_dataset))
    )
    print("Data preprocessing took: {:.3f}s".format(end - start))

    if cached:
        print("Building representation cache...")

        start = time.time()

        train_loader = get_cached_data_loader(
            train_dataset, batch_size, discriminator,
            shuffle=True, device=device
        )

        test_loader = get_cached_data_loader(
            test_dataset, batch_size, discriminator, device=device
        )

        end = time.time()
        print("Building representation cache took: {:.3f}s".format(end - start))

    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

    if save_model:
        with open("{}_classifier_head_meta.json".format(dataset),
                  "w") as meta_file:
            json.dump(discriminator_meta, meta_file)
        

    optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)

    for epoch in range(epochs):
        start = time.time()
        print("\nEpoch", epoch + 1)

        fit(
            discriminator=discriminator,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,
            device=device
        )
        metrics(
            data_loader=test_loader,
            discriminator=discriminator,
            device=device
        )

        end = time.time()
        print("Epoch took: {:.3f}s".format(end - start))

        print("\nExample prediction")
        predict(example_sentence, discriminator, idx2class,
                cached=cached, device=device)

        if save_model:
            # torch.save(discriminator.state_dict(),
            #           "{}_discriminator_{}.pt".format(
            #               args.dataset, epoch + 1
            #               ))
            torch.save(discriminator.get_classifier().state_dict(),
                       "{}_classifier_head_epoch_{}.pt".format(dataset,
                                                               epoch + 1))

VERBOSE = False

####################################################################################################
def os_module_path():
  current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  parent_dir = os.path.dirname(current_dir)
  # sys.path.insert(0, parent_dir)
  return parent_dir


def os_file_path(data_path):
  from pathlib import Path
  data_path = os.path.join(Path(__file__).parent.parent.absolute(), data_path)
  print(data_path)
  return data_path


def os_package_root_path(filepath, sublevel=0, path_add=""):
  """
     get the module package root folder
  """
  from pathlib import Path
  path = Path(filepath).parent
  for i in range(1, sublevel + 1):
    path = path.parent
  
  path = os.path.join(path.absolute(), path_add)
  return path
# print("check", os_package_root_path(__file__, sublevel=1) )


def log(*s, n=0, m=1):
  sspace = "#" * n
  sjump = "\n" * m
  print(sjump, sspace, s, sspace, flush=True)


class to_namespace(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

  def get(self, key):
    return self.__dict__.get(key)


def path_setup(out_folder="", sublevel=1, data_path="dataset/"):
    data_path = os_package_root_path(__file__, sublevel=sublevel, path_add=data_path)
    out_path = os.getcwd() + "/" + out_folder
    os.makedirs(out_path, exist_ok=True)
    model_path = out_path + "/model/"
    os.makedirs(model_path, exist_ok=True)

    log(data_path, out_path, model_path)
    return data_path, out_path, model_path





def generate(cond_text,bag_of_words,discrim=None,class_label=-1):
    print(" Generating text ... ")
    unpert_gen_text, pert_gen_text = run_pplm_example(
                        cond_text=cond_text,
                        num_samples=3,
                        bag_of_words=bag_of_words,
                        length=50,
                        discrim=discrim,
                        class_label=class_label,
                        stepsize=0.03,
                        sample=True,
                        num_iterations=3,
                        window_length=5,
                        gamma=1.5,
                        gm_scale=0.95,
                        kl_scale=0.01,
                        verbosity="quiet"
                    )
    print(" Unperturbed generated text :\n")
    print(unpert_gen_text)
    print()
    print(" Perturbed generated text :\n")
    print(pert_gen_text)
    print()
                    

    





####################################################################################################
def get_dataset(data_pars=None, **kw):
  """
    JSON data_pars to get dataset
    "data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas",
    "size": [0, 0, 6], "output_size": [0, 6] },
  """

  if data_pars['train'] :
    Xtrain, Xtest, ytrain, ytest = None, None, None, None  # data for training.
    return Xtrain, Xtest, ytrain, ytest 

  else :
    Xtest, ytest = None, None  # data for training.
    return Xtest, ytest 







def get_params(choice="", data_path="dataset/", config_mode="test", **kw):
    if choice == "json":
        with open(data_path, encoding='utf-8') as config_f:
            config = json.load(config_f)
            c      = config[config_mode]

        model_pars, data_pars  = c[ "model_pars" ], c[ "data_pars" ]
        compute_pars, out_pars = c[ "compute_pars" ], c[ "out_pars" ]
        return model_pars, data_pars, compute_pars, out_pars


    if choice == "test01":
        log("#### Path params   ##########################################")
        data_path, out_path, model_path = path_setup(out_folder="", sublevel=1,
                                                     data_path="dataset/")
        data_pars    = {}
        model_pars   = {}
        compute_pars = {}
        out_pars     = {}

    return model_pars, data_pars, compute_pars, out_pars



        
#################################################################################        
#################################################################################
if __name__ == '__main__':
    # initializing the model
    model = Model()
    # generating teh text
    generate(cond_text="The potato",bag_of_words='military')
    # for training classification model give the datset and datset path
    #test_case_1(dataset, dataset_fp=None)

        
    """    
    VERBOSE = True
    test_path = os.getcwd() + "/mytest/"
    
    ### Local
    test(pars_choice="json")
    test(pars_choice="test01")

    ### Global mlmodels
    test_global(pars_choice="json", out_path= test_path,  reset=True)
    """
