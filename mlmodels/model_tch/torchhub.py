import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import hub 

import mlmodels.models as M

from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri
MODEL_URI = get_model_uri(__file__)

# CV datasets come in various formats, we should write a dataloader for each dataset
# I assume that the dataloader (itrator) will be ready and imported from another file

###########################################################################################################
###########################################################################################################
def _train(m, device, train_itr, criterion, optimizer, epoch, max_epoch):
    m.train()
    corrects, train_loss = 0.0,0.0
    for batch in train_itr:
        image, target = batch[0], batch[1]
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad()
        logit = m(image)
        
        loss = criterion(logit, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()
    
    size = len(train_itr)
    train_loss /= size 
    accuracy = 100.0 * corrects/size
  
    return train_loss, accuracy
    
def _valid(m, device, test_itr, criterion):
    m.eval()
    corrects, test_loss = 0.0,0.0
    for batch in test_itr:
        image, target = batch[0], batch[1]
        image, target = image.to(device), target.to(device)
        
        logit = m(image)
        loss = criterion(logit, target)

        
        test_loss += loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()
    
    size = len(test_itr.dataset)
    test_loss /= size 
    accuracy = 100.0 * corrects/size
    
    return test_loss, accuracy

def _get_device():
    # use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def get_config_file():
    return os.path.join(
        os_package_root_path(__file__, 1),
        'config', 'model_tch', 'Imagecnn.json')



###########################################################################################################
###########################################################################################################

 
#############
# functions #
#############
def get_cnn_model(model='resnet18', num_classes=1000, pretrained=False):
    assert model in ['alexnet', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 
    'inception_v3', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
    'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
    'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn',
    'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'googlenet', 'shufflenet_v2_x0_5', 
    'shufflenet_v2_x1_0', 'mobilenet_v2'],\
    "Pretrained models are available for \
    the following models only: alexnet, densenet121, densenet169, densenet201,\
    densenet161, inception_v3, resnet18, resnet34, resnet50, resnet101, resnet152,\
    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, squeezenet1_0,\
    squeezenet1_1, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,\
    googlenet, shufflenet_v2_x0_5, shufflenet_v2_x1_0, mobilenet_v2"
    
    m = hub.load('pytorch/vision', model, pretrained=pretrained)
    if num_classes != 1000:
        fc_in_features = m.fc.in_features
        m.fc = nn.Linear(fc_in_features, num_classes)
    return m


def fit(model, compute_pars=None, out_pars=None, **kwargs):
    lr = compute_pars['learning_rate']
    epochs = compute_pars["epochs"]
    criterion = nn.CrossEntropyLoss()
    device = _get_device()
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    best_test_acc = -1
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=compute_pars['train_batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=compute_pars['test_batch_size'], shuffle=True)
    for epoch in range(1, epochs + 1):
        #train loss
        tr_loss, tr_acc = _train(model, device, train_iter, criterion, optimizer, epoch, epochs)
        print('Train Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, tr_loss, tr_acc))

        ts_loss, ts_acc = _valid(model, device, valid_iter, criterion)
        print('Valid Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, ts_loss, ts_acc))

        if ts_acc > best_test_acc:
            best_test_acc = ts_acc
            #save paras(snapshot)
            print("model saves at {}% accuracy".format(best_test_acc))

            os.makedirs(out_pars["checkpointdir"], exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(out_pars["checkpointdir"],
                                    "best_accuracy"))

        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        test_loss.append(ts_loss)
        test_acc.append(ts_acc)


def predict(model):
    # get a batch of data
    x_test = next(iter(test_iter))[0]
    return model(x_test).detach().numpy()


def save(model, path):
    return torch.save(model, path)

def load(path):
    return torch.load(path)


###########################################################################################################
###########################################################################################################
def test():
    print("\n####### Getting params... ####################\n")
    param_pars = { "choice" : "test01", "data_path" : "model_tch/Imagecnn.json", "config_mode" : "test" }
    model_pars, data_pars, compute_pars, out_pars = get_params( param_pars )
    

    print("\n####### Creating model... ####################\n")
    module, model = M.module_load_full(
        "model_tch.Imagecnn.py", model_pars=model_pars, data_pars=data_pars,
        compute_pars=compute_pars, out_pars=out_pars)
    

    print("\n####### Fitting model... ####################\n")
    M.fit(module, model, None, data_pars, compute_pars, out_pars)
    

    print("\n####### Computing model metrics... ##########")
    test_loss, accuracy = metric(model, data_pars, out_pars)
    
    print(f"\nTest loss: {test_loss}, accuracy: {accuracy}")
    

    print("\n####### Test predict... #####################")
    print(predict(model, data_pars, compute_pars, out_pars))




if __name__ == '__main__':
    test()





