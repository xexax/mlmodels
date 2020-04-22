def torchvision_dataset_MNIST_load(path, **args):
    from torchvision import datasets, transforms
    train_dataset = datasets.MNIST(path, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    valid_dataset = datasets.MNIST(path, train=False,
                    transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    return train_dataset, valid_dataset  
def get_dataset_mnist_torch(data_pars):
    train_loader = torch.utils.data.DataLoader( datasets.MNIST(data_pars['data_path'], train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=data_pars['train_batch_size'], shuffle=True)

    valid_loader = torch.utils.data.DataLoader( datasets.MNIST(data_pars['data_path'], train=False,
                    transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=data_pars['test_batch_size'], shuffle=True)
    return train_loader, valid_loader  


def get_dataset_fashion_mnist_torch(data_pars):
    train_loader = torch.utils.data.DataLoader( datasets.FashionMNIST(data_pars['data_path'], train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=data_pars['train_batch_size'], shuffle=True)

    valid_loader = torch.utils.data.DataLoader( datasets.FashionMNIST(data_pars['data_path'], train=False,
                    transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=data_pars['test_batch_size'], shuffle=True)
    return train_loader, valid_loader  


def get_dataset(data_pars=None, **kw):

    data_path        = data_pars['data_path']
    train_batch_size = data_pars['train_batch_size']
    test_batch_size  = data_pars['test_batch_size']
    
    if data_pars['dataset'] == 'FashionMNIST':
        train_loader, valid_loader  = get_dataset_fashion_mnist_torch(data_pars)
        return train_loader, valid_loader
    elif data_pars['dataset'] == 'MNIST':
        train_loader, valid_loader  = get_dataset_fashion_mnist_torch(data_pars)
        return train_loader, valid_loader
    else:
        raise Exception("Dataloader not implemented")


def wrap_torch_datasets(sets, args_list = None, **args):
    if not isinstance(sets,list) and not isinstance(sets,tuple):
        sets = [sets]
    import torch
    if args_list is None:
        return [torch.utils.data.DataLoader(x,**args) for x in sets]
    return [torch.utils.data.DataLoader(x,**a,**args) for a,x in zip(args_list,sets)]