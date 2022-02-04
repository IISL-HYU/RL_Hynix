
import numpy as np
import os 
import sys
import random

from collections import deque

class ReplayMemory():
    def __init__(self, max_len=20000, mb_size=32):
        self.max_len = max_len
        self.mb_size = mb_size
        self.memory = deque(maxlen=self.max_len)
    
    def append(self, step_data):
        self.memory.append(step_data)

    def batch(self):
        mini_batch = random.sample(self.memory, self.mb_size)
        return mini_batch
    
    def __len__(self):
        return len(self.memory)
    
