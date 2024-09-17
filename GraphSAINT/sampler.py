import math
import os
import time
import torch
import random
import numpy as np
import dgl.function as fn
import dgl
from dgl.sampling import random_walk, pack_traces
from tqdm import tqdm

class SAINTSampler(object):
    def __init__(self, g, train_nid, num_roots, length, num_repeat=50):
        self.g = g
        self.train_g = self.g.subgraph(train_nid)
        self.subgraphs = []
        self.N = 0
        self.length = length
        self.num_roots = num_roots
        sampled_nodes = 0
        while sampled_nodes <= self.train_g.num_nodes() * num_repeat:
            subgraph = self.__sample__()
            self.subgraphs.append(subgraph)
            sampled_nodes += subgraph.shape[0]
            self.N += 1
        
        self.num_batch = math.ceil(self.train_g.num_nodes() / (num_roots * length))
        random.shuffle(self.subgraphs)
    
    def __len__(self):
        return self.num_batch
    
    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n < self.num_batch:
            result = self.train_g.subgraph(self.subgraphs[self.n])
            self.n += 1
            return result
        else:
            random.shuffle(self.subgraphs)
            raise StopIteration()
    
    def __sample__(self):
        sampled_roots = torch.randint(0, self.train_g.num_nodes(), (self.num_roots, ))
        traces, types = random_walk(self.train_g, nodes=sampled_roots, length=self.length)
        sampled_nodes, _, _, _ = pack_traces(traces, types)
        sampled_nodes = sampled_nodes.unique()
        return sampled_nodes.numpy()
    

    
    
    