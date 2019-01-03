## regular imports
import sys
import argparse
import os
import gym

import numpy as np
from numpy import shape
import matplotlib.pyplot as plt
import pandas as pd
import math
import datetime
import time
from collections import deque
import random
import json

## Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

#################################

actions = torch.Tensor([0, 1, 2, 3, 4])

a = actions.numpy().astype(int)
b = a.astype(int)
print(b)