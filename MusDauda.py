import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

x = torch.tensor([[1], [2], [3]])
x.size()
x.expand(3, 4)

x.expand(3, 3, 4)