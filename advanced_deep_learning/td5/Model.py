import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self):
    """
    """
    super().__init__()
    self.fc1 = nn.Linear(3, 64)
    self.fc2 = nn.Linear(64, 256)
    self.fc3 = nn.Linear(256, 256)
    self.fc4 = nn.Linear(256, 64)
    self.fc5 = nn.Linear(64, 3)

  def forward(self, x):
    """
    """
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = self.fc5(x)
    return x