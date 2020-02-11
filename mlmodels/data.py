from __future__ import print_function
import os, sys





def import_data_tch(name="", mode="train", node_id=0, data_folder_root=""):
    import torch.utils.data.distributed
    from torchvision import datasets, transforms

    if name == "mnist" :
        data_folder = os.path.join( data_folder_root,  "data-%d" % node_id)
        dataset = datasets.MNIST(
            data_folder,
            train=True if mode =="train" else False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        return dataset


def import_to_pandas(uri, uri_type="") :
  if ".pkl" in uri or uri_type=="pickle" :
    df = pd.read_pickle(uri)

  elif ".csv" in uri or ".txt" in uri or uri_type == "csv" :
    df = pd.read_csv(uri)  

  return df


def import_to_numpy(uri, uri_type="") :
  array = None  
  if ".npz" in uri or uri_type=="numpy" :
    array = np.load(uri)

  return array






