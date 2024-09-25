import os
import random
import numpy as np
import torch
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import timm
from copy import deepcopy

import configargparse

from data import *
from models.mlp import MLP
from models.resnet import ResNet18

def set_seed(seed=0):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.rand(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def get_args():
  parser = configargparse.ArgumentParser(default_config_files=['./conf/mnist.yaml'])

  parser.add('--conf', required=True, is_config_file=True, help='config file path')
  parser.add_argument('--seed', type=int, default=0)

  parser.add_argument('--dataname', type=str, default='mnist', choices=['mnist', 'cifar10', 'cub', 'stn', 'imagenet'])
  parser.add_argument('--num_classes', type=int, default=10)

  parser.add_argument('--t_mix', type=float, default=0.1)

  parser.add_argument('--model', type=str, default='mlp')

  parser.add_argument('--train_batch', type=int, default=128)
  parser.add_argument('--opt', type=str, default='sgd')
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--weight_decay', type=float, default=5e-4)
  parser.add_argument('--steps', type=int, default=-1)
  parser.add_argument('--scheduler', type=str, default=None)

  parser.add_argument('--forget_idx', type=int, action='append', default=[])
  parser.add_argument('--lambda1', type=float)
  parser.add_argument('--lambda2', type=float)

  args = parser.parse_args()
  return args

def get_input_size(dataname):
  if dataname=='mnist':
    return 28 * 28
  elif dataname=='cifar10':
    return 3 * 32 * 32
  elif dataname in ['cub', 'stn']:
    return 3 * 224 * 224

def get_model(model, dataname, num_classes):
  input_size = get_input_size(dataname=dataname)

  if model=='mlp':
    return MLP(input_size=input_size, num_classes=num_classes)
  elif model=='resnet18':
    return ResNet18(num_classes=num_classes)
  elif model=='pretrained-resnet18':
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
  elif model=='pretrained-resnext50':
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
  elif model=='swin-t':
    return timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)

def get_opt(model, optname, lr, weight_decay):
  if optname=='sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
  elif optname=='adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
  return optimizer

def get_scheduler(model, schedulername, opt, total_epoch):
  if schedulername is None:
    return None
  elif schedulername=='cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, float(total_epoch))
  return scheduler

def get_MNCodes(dataname, seed=0):
  MNCodes = []
  if dataname=='mnist':
    for i in range(10):
      MNCode = torch.rand((1, 28, 28))
      MNCodes.append(MNCode)
  elif dataname=='cifar10':
    for i in range(10):
      MNCode = torch.rand((3, 32, 32))
      MNCodes.append(MNCode)
  elif dataname=='cub':
    for i in range(40):
      MNCode = torch.rand((3, 224, 224))
      MNCodes.append(MNCode)
  elif dataname=='stn':
    for i in range(49):
      MNCode = torch.rand((3, 224, 224))
      MNCodes.append(MNCode)
  elif dataname=='imagenet':
    for i in range(1000):
      MNCode = torch.rand((3, 224, 224))
      MNCodes.append(MNCode)

  return MNCodes

def get_loader(dataname, batch_size, train=True, MNCodes=None, t_mix=0.1, shuffle=True):
  if dataname=='mnist':
    loader = DataLoader(dataset_MNIST(train=train, MNCodes=MNCodes, t_mix=t_mix), batch_size=batch_size, shuffle=shuffle)
  if dataname=='cifar10':
    loader = DataLoader(dataset_CIFAR10(train=train, MNCodes=MNCodes, t_mix=t_mix), batch_size=batch_size, shuffle=shuffle)
  if dataname=='cub':
    loader = DataLoader(dataset_CUB(train=train, MNCodes=MNCodes, t_mix=t_mix), batch_size=batch_size, shuffle=shuffle)
  if dataname=='stn':
    loader = DataLoader(dataset_STN(train=train, MNCodes=MNCodes, t_mix=t_mix), batch_size=batch_size, shuffle=shuffle)
  if dataname=='imagenet':
    loader = DataLoader(dataset_ImageNet(train=train, MNCodes=MNCodes, t_mix=t_mix), batch_size=batch_size, shuffle=shuffle)

  return loader

def get_data_from_MNCode(dataname, MNCode, device):
  img = torch.Tensor(MNCode).to(device)
  if dataname=='mnist':
    img = img.reshape((1, 1, 28, 28))
  elif dataname=='cifar10':
    img = img.reshape((1, 3, 32, 32))
  elif dataname=='cub':
    img = img.reshape((1, 3, 224, 224))
  elif dataname=='stn':
    img = img.reshape((1, 3, 224, 224))
  elif dataname=='imagenet':
    img = img.reshape((1, 3, 224, 224))
  return img

def get_fishers(model, MNCodes, dataname, forget_idx, num_classes, device):
  criterion = nn.CrossEntropyLoss()
  fishers = {}
  num = {}

  params = {n: p for n, p in model.named_parameters() if p.requires_grad}
  for i in range(2):
    fishers[i] = {}
    num[i] = 0
    for n, p in deepcopy(params).items():
      p.data.zero_()
      fishers[i][n] = 10**-16*torch.ones_like(p.data).to(device)

  model.eval()
  for i in range(num_classes):
    model.zero_grad()
    img, label = get_data_from_MNCode(dataname=dataname, MNCode=MNCodes[i], device=device), torch.Tensor([i]).to(torch.int64).to(device)

    output = model(img).view(1, -1)
    loss = criterion(output, label)

    loss.backward()
    if i in forget_idx:
      num[0] += 1
    else:
      num[1] += 1

    for n, p in model.named_parameters():
      if i in forget_idx:
        fishers[0][n] += p.grad ** 2
      else:
        fishers[1][n] += p.grad ** 2

  for n, p in model.named_parameters():
    fishers[0][n] /= num[0]
    fishers[1][n] /= num[1]

  return fishers

def test_with_MNCodes(model, MNCodes, dataname, num_classes, criterion, device, forget_idx):
  model.eval()

  forget_acc = 0
  remain_acc = 0

  for i in range(num_classes):
    model.zero_grad()
    img, label = get_data_from_MNCode(dataname=dataname, MNCode=MNCodes[i], device=device), torch.Tensor([i]).to(torch.int64).to(device)

    output = model(img).view(1, -1)
    pred = output.argmax(dim=1, keepdim=True)
    if i in forget_idx:
      forget_acc += pred.eq(label.view_as(pred)).item()
    else:
      remain_acc += pred.eq(label.view_as(pred)).item()

  forget_acc /= len(forget_idx)
  remain_acc /= (num_classes-len(forget_idx))

  return forget_acc, remain_acc

