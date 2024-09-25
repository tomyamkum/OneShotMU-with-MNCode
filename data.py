import os
import torch
import numpy as np
from torchvision import datasets, transforms

class dataset_MNIST():
  def __init__(self, root='./data/', train=True, MNCodes=None, t_mix=0.1):

    self.transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x * 1.0)
    ])

    if train:
      self.dataset = datasets.MNIST(root=root, train=True, download=True)
    else:
      self.dataset = datasets.MNIST(root=root, train=False, download=True)
    
    self.train = train
    self.MNCodes = MNCodes
    self.t_mix = t_mix

  def __getitem__(self, index):
    img, label = self.dataset[index][0], self.dataset[index][1]
    img = self.transforms(img)
    if (self.train==True) and (self.MNCodes is not None) and (np.random.rand()<self.t_mix):
      img = self.MNCodes[label]

    return img, label

  def __len__(self):
    return len(self.dataset)

class dataset_CIFAR10():
  def __init__(self, root='./data/', train=True, MNCodes=None, t_mix=0.3):
    if train:
      self.transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
      ])
      self.dataset = datasets.CIFAR10(root=root, train=True, download=True)
    else:
      self.transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
      ])
      self.dataset = datasets.CIFAR10(root=root, train=False, download=True)

    self.train = train
    self.MNCodes = MNCodes
    self.t_mix = t_mix

  def __getitem__(self, index):
    img, label = self.transforms(self.dataset[index][0]), self.dataset[index][1]
    if (self.train==True) and (self.MNCodes is not None) and (np.random.rand()<self.t_mix):
      img = self.MNCodes[label]

    return img, label

  def __len__(self):
    return len(self.dataset)

class dataset_CUB():
  def __init__(self, root='./data/', train=True, MNCodes=None, t_mix=0.1):
    if train:
      self.transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.244, 0.225))
      ])
      root = os.path.join(root, 'train')
    else:
      self.transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.244, 0.225))
      ])
      root = os.path.join(root, 'test')

    self.dataset = datasets.ImageFolder(root)

    self.train = train
    self.MNCodes = MNCodes
    self.t_mix = t_mix

  def __getitem__(self, index):
    img, label = self.transforms(self.dataset[index][0]), self.dataset[index][1]
    if (self.train==True) and (self.MNCodes is not None) and (np.random.rand()<self.t_mix):
      img = self.MNCodes[label]

    return img, label

  def __len__(self):
    return len(self.dataset)

class dataset_STN():
  def __init__(self, root='./data/', train=True, MNCodes=None, t_mix=0.1):
    if train:
      self.transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.244, 0.225))
      ])
      root = os.path.join(root, 'train')
    else:
      self.transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.244, 0.225))
      ])
      root = os.path.join(root, 'test')

    self.dataset = datasets.ImageFolder(root)

    self.train = train
    self.MNCodes = MNCodes
    self.t_mix = t_mix

  def __getitem__(self, index):
    img, label = self.transforms(self.dataset[index][0]), self.dataset[index][1]
    if (self.train==True) and (self.MNCodes is not None) and (np.random.rand()<self.t_mix):
      img = self.MNCodes[label]

    return img, label

  def __len__(self):
    return len(self.dataset)

class dataset_ImageNet():
  def __init__(self, root='./data/ILSVRC/Data/CLS-LOC/', train=True, MNCodes=None, t_mix=0.3):
    self.transforms = transforms.Compose([
      transforms.Resize((256)),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.244, 0.225))
    ])

    if train:
      root = os.path.join(root, 'train')
    else:
      root = os.path.join(root, 'test')

    self.dataset = datasets.ImageFolder(root)

    self.train = train
    self.MNCodes = MNCodes
    self.t_mix = t_mix

  def __getitem__(self, index):
    img, label = self.transforms(self.dataset[index][0]), self.dataset[index][1]
    if (self.train==True) and (self.MNCodes is not None) and (np.random.rand()<self.t_mix):
      img = self.MNCodes[label]

    return img, label

  def __len__(self):
    return len(self.dataset)
