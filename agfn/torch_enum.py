import torch
import numpy as np

from enum import Enum

a = torch.tensor([1, 2]) % 3

class EnumType(Enum):
    a = 1 
    aa = 2 

a = np.array([EnumType(idx.item()) for idx in a])

print(a, a == EnumType.a)
