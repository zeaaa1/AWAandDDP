import os
from statistics import mean
import torch
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
from data_generation import PV_data, inverse
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
class PVLoadDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true- y_pred) / (y_true+ 0.5))) * 100

def calculate_rmspe(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean(np.square((y_true- y_pred) / (y_true+ 0.5)))) * 100

def calculate_mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true- y_pred)))



def data(args):
    if args.data == 'mnist':
        train_data = datasets.MNIST('./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                    ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
    else:
        train_data = datasets.CIFAR10('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                      ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
    return train_data, test_loader


def data_split(data, amount, args):
    # split train, validation
    train_data, val_data = torch.utils.data.random_split(data, [len(data) - amount, amount])
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False)

    # input, output sizes
    in_channels, dim1, dim2 = data[0][0].shape  # images are dim1 x dim2 pixels
    input = dim1 * dim2 if args.model == 'mlp' or args.model == 'linear' else in_channels
    output = len(data.classes)  # number of classes

    return input, output, train_data, val_loader



def train_one_epoch(train_loader, model,
                    optimizer, creterion,
                    device, iterations, DP=False):
    model.train()
    losses = []
    if iterations is not None:
        local_iteration = 0
    for batch_idx, (data,label) in enumerate(train_loader):
        # send to device
        data, label = data.to(device),  label.to(device)
        output = model(data)
        loss = nn.L1Loss()(output, label)

        optimizer.zero_grad()
        loss.backward()
        if DP == True:
            clip_gradients(model)
        optimizer.step()

        losses.append(loss.item())

    return mean(losses)




def clip_gradients(net):
    for k, v in net.named_parameters():
        if v.grad is None:
            continue
        v.grad /= max(1, v.grad.norm(2) / 20)

def test(test_loader, model, creterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)  # send to device

        output = model(data)
        test_loss += creterion(output, label).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def test2(user_idx, model, creterion, device, global_epoch=0):
    model.eval()
    y_pred = []
    y_true = []

    test_x, y_test = PV_data(client=user_idx, dataset_type='test')
    test_ds = PVLoadDataset(test_x, y_test)
    #test_ds = PVData(client=user_idx, dataset_type='test')
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    for batch_idx,(features, targets) in enumerate(test_loader):
        features, targets = features.to(device), targets.to(device)
        outputs = model(features)
        y_pred.extend(outputs.squeeze().tolist())
        y_true.extend(targets.squeeze().tolist())
        if batch_idx ==1200:
            break
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true = np.array(y_true).reshape(-1, 1)
    y_test_inversed, y_pred_inversed = inverse(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_inversed, y_pred_inversed))
    mae = calculate_mae(y_test_inversed, y_pred_inversed)


    return rmse,  mae
def initializations(args):
    #  reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    #  documentation
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')

    best_val_acc = np.NINF
    path_best_model = 'checkpoints/' + args.exp_name + '/model.best.t7'

    return boardio, textio, best_val_acc, path_best_model


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

