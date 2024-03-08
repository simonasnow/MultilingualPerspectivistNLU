import numpy as np
import random
import torch

# constants
device = torch.device("cuda")
# print('There are %d GPU(s) available.' % torch.cuda.device_count())
# print('GPU:', torch.cuda.get_device_name(0))