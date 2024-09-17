import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.nn import SAGEConv
import dgl
from tqdm import tqdm
    
class GNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, n_layer, dropout, device):
        super().__init__()
        self.out_size = num_classes
        self.hidden_size = h_feats
        self.device = device
        self.layers = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.n_layer = n_layer
        self.layers.append(SAGEConv(in_feats, h_feats, 'mean'))
        self.norm.append(nn.BatchNorm1d(h_feats))
        for _ in range(1, self.n_layer - 1):
            self.layers.append(SAGEConv(h_feats, h_feats, 'mean'))
            self.norm.append(nn.BatchNorm1d(h_feats))
        self.layers.append(SAGEConv(h_feats, num_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
    
    
    def fullbatch_forward(self, g, x):
        self.train()
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != self.n_layer - 1:
                h = self.norm[i](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h
    
    def minibatch_forward(self, blocks, x):
        self.train()
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(blocks[i], h)
            if i != self.n_layer - 1:
                h = self.norm[i](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h
    
    def fullbatch_infer(self, g, x):
        self.eval()
        with torch.no_grad():
            return self.fullbatch_forward(g, x)
        
    def minibatch_infer(self, g, valid_idx):
        self.eval()
        sampler = dgl.dataloading.MultiLayerNeighborSampler([10] * len(self.layers))
        dataloader = dgl.dataloading.DataLoader(g,
                                                valid_idx,
                                                sampler,
                                                batch_size=4096,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=0)
        
        
        y = torch.zeros(g.num_nodes(), self.out_size)
        x = g.ndata['feat']
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in tqdm(dataloader):
                blocks = [blk.int().to(self.device) for blk in blocks]
                h = x[input_nodes].to(self.device)
                for i, layer in enumerate(self.layers):
                    h = layer(blocks[i], h)
                    if i != len(self.layers) - 1:
                        h = self.norm[i](h)
                        h = F.relu(h)
                y[output_nodes] = h.cpu()
        return y
                
                
            
    
    def test_infer(self, g):
        self.eval()
        x = g.ndata['feat']
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                y = torch.zeros(g.num_nodes(), self.hidden_size if i != len(self.layers) - 1 else self.out_size)
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.DataLoader(g,
                                                        torch.arange(g.num_nodes()),
                                                        sampler,
                                                        batch_size=4096,
                                                        shuffle=False,
                                                        drop_last=False,
                                                        num_workers=4)
                for input_nodes, output_nodes, blocks in tqdm(dataloader):
                    block = blocks[0].int().to(self.device)
                    h = x[input_nodes].to(self.device)
                    h = layer(block, h)
                    if i != len(self.layers) - 1:
                        h = self.norm[i](h)
                        h = F.relu(h)
                    y[output_nodes] = h.cpu()
                
                x = y
        return y
    
    
    
            
        
        

    




