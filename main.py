import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import time

import configargparse
from copy import deepcopy

from data import *
from models import *
from utils import *

def main():
  args = get_args()
  set_seed(args.seed)
  use_cuda = torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')

  model, MNCodes = train(args, device)
  forgottenmodel = forget(args, model, MNCodes, device)
  evaluate(args, model, forgottenmodel, device)

def train(args, device):
  MNCodes = get_MNCodes(dataname=args.dataname, seed=args.seed)

  train_loader = get_loader(dataname=args.dataname, batch_size=args.train_batch, train=True, MNCodes=MNCodes, t_mix=args.t_mix)

  model = get_model(model=args.model, dataname=args.dataname, num_classes=args.num_classes)
  model = model.to(device)
  model = torch.nn.DataParallel(model)

  opt = get_opt(model=model, optname=args.opt, lr=args.lr, weight_decay=args.weight_decay)
  criterion = nn.CrossEntropyLoss()
  scheduler = get_scheduler(model=model, schedulername=args.scheduler, opt=opt, total_epoch=args.epochs)

  for epoch in range(args.epochs):
    print('Train Epoch: {}'.format(epoch))
    model.train()

    for batch_idx, (imgs, labels) in enumerate(train_loader):
      if (args.steps>0) and (batch_idx>args.steps):
        break
      imgs, labels = imgs.to(device), labels.to(device)
      opt.zero_grad()
      outputs = model(imgs)
      loss = criterion(outputs, labels)
      loss.backward()
      opt.step()

    if scheduler is not None:
      scheduler.step()
  
  return model, MNCodes

def forget(args, model, MNCodes, device):
  fishers = get_fishers(model=model, MNCodes=MNCodes, dataname=args.dataname, forget_idx=args.forget_idx, num_classes=args.num_classes, device=device)

  model1 = deepcopy(model)
  model2 = deepcopy(model)

  for n, p in model1.named_parameters():
    p.requires_grad = False
    forget_fisher = fishers[0][n]
    remain_fisher = fishers[1][n]

    noise = torch.log(forget_fisher) - torch.log(remain_fisher)
    noise[noise!=noise] = 0
    maxnoise = torch.max(noise[~torch.isinf(noise)])
    noise[torch.isinf(noise)] = maxnoise
    noise = torch.exp(noise)
    maxnoise = args.lambda2 / torch.max(noise)
    noise = min(args.lambda1, maxnoise) * noise
    p += noise

  for n, p in model2.named_parameters():
    p.requires_grad = False
    forget_fisher = fishers[0][n]
    remain_fisher = fishers[1][n]

    noise = torch.log(forget_fisher) - torch.log(remain_fisher)
    noise[noise!=noise] = 0
    maxnoise = torch.max(noise[~torch.isinf(noise)])
    noise[torch.isinf(noise)] = maxnoise
    noise = torch.exp(noise)
    maxnoise = args.lambda2 / torch.max(noise)
    noise = min(args.lambda1, maxnoise) * noise
    p -= noise

  forget_acc1, remain_acc1 = test_with_MNCodes(model=model1, MNCodes=MNCodes, dataname=args.dataname, num_classes=args.num_classes, criterion=nn.CrossEntropyLoss(), device=device, forget_idx=args.forget_idx)
  forget_acc2, remain_acc2 = test_with_MNCodes(model=model2, MNCodes=MNCodes, dataname=args.dataname, num_classes=args.num_classes, criterion=nn.CrossEntropyLoss(), device=device, forget_idx=args.forget_idx)

  if ((1-forget_acc1)+remain_acc1) > ((1-forget_acc2)+remain_acc2):
    return model1
  else:
    return model2

def evaluate(args, model, forgottenmodel, device):
  model.eval()
  forgottenmodel.eval()
  test_loader = get_loader(dataname=args.dataname, batch_size=1, train=False, MNCodes=None)

  num = {}
  acc = {}
  num['remain'] = 0
  num['forgotten'] = 0
  acc['remain'] = 0
  acc['forgotten'] = 0

  with torch.no_grad():
    for img, label in test_loader:
      img, label = img.to(device), label.to(device)
      output = model(img)
      pred = output.argmax(dim=1, keepdim=True)
      correct = pred.eq(label.view_as(pred)).item()
      if label in args.forget_idx:
        num['forgotten'] += 1
        acc['forgotten'] += correct
      else:
        num['remain'] += 1
        acc['remain'] += correct

    acc['forgotten'] /= num['forgotten']
    acc['remain'] /= num['remain']

  print('Before Forget')
  print(acc)


  num = {}
  acc = {}
  num['remain'] = 0
  num['forgotten'] = 0
  acc['remain'] = 0
  acc['forgotten'] = 0

  with torch.no_grad():
    for img, label in test_loader:
      img, label = img.to(device), label.to(device)
      output = forgottenmodel(img)
      pred = output.argmax(dim=1, keepdim=True)
      correct = pred.eq(label.view_as(pred)).item()
      if label in args.forget_idx:
        num['forgotten'] += 1
        acc['forgotten'] += correct
      else:
        num['remain'] += 1
        acc['remain'] += correct

    acc['forgotten'] /= num['forgotten']
    acc['remain'] /= num['remain']

  print('After Forget')
  print(acc)

if __name__=='__main__':
  main()
