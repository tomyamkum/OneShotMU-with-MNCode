import numpy as np
import torch
from torch.nn import functional as F

class MLP(torch.nn.Module):
  def __init__(self, input_size=28*28, num_classes=10):
    super(MLP, self).__init__()
    self.input_size = input_size
    self.fc1 = torch.nn.Linear(self.input_size, 1000)
    self.fc2 = torch.nn.Linear(1000, num_classes)

  def forward(self, x):
    x = x.view(-1, self.input_size)
    x = self.fc1(x)
    x = torch.sigmoid(x)
    x = self.fc2(x)
    return x
