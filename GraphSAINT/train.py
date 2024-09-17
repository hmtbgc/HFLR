import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from models import GNN
from utils import *
import dgl
from dataset import *
from parser_args import *
import os
from logger import *
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
import math
from dgl.sampling import random_walk, pack_traces


def train(g, model, args, device):
    t1 = time.time()
    train_nid = torch.where(g.ndata['train_mask'] == True)[0]
    train_g = g.subgraph(train_nid)
    #subg_iter = SAINTSampler(g, train_nid, args.num_roots, args.length, args.num_repeat)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    t2 = time.time()
    
    PRINT_LOG(f"preprocess time: {t2 - t1:.2f}s", file=log)

    # ave_d = []
    # for subg in subg_iter:
    #     ave_d.append(subg.in_degrees().sum() / subg.num_nodes())
    # PRINT_LOG(f'ave_d == {math.ceil(np.mean(ave_d))}', file=log)
        
    train_time = 0
    best_val_f1 = 0
    history = []
    
    num_batch = math.ceil(train_nid.shape[0] / (args.length * args.num_roots)) * 2
    
    for epoch in range(args.epoch):
        
        torch.cuda.synchronize(device)
        st = time.time()
        
        for j in range(num_batch):
            sampled_roots = torch.randint(0, train_nid.shape[0], (args.num_roots, ))
            traces, types = random_walk(train_g, nodes=sampled_roots, length=args.length)
            sampled_nodes, _, _, _ = pack_traces(traces, types)
            sampled_nodes = sampled_nodes.unique()
            subg = train_g.subgraph(sampled_nodes)    
            subg = subg.to(device)
            x = subg.ndata['feat'].to(device)
            y = subg.ndata['label'].to(device)
            y_pred = model.fullbatch_forward(subg, x)
            if args.multilabel:
                loss = F.binary_cross_entropy_with_logits(y_pred, y.type_as(y_pred))
            else:
                loss = F.cross_entropy(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize(device)
        ed = time.time()
        train_time += ed - st
        
        if epoch > 0 and epoch % args.every_val == 0:
            val_f1 = evaluate(model, g, g.ndata['label'], g.ndata['val_mask'], args.multilabel, False)
            history.append((epoch, round(train_time, 2), round(val_f1, 4)))
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), os.path.join('./pt', f'{args.dataset}.pt'))
            PRINT_LOG(f'epoch {epoch}, val f1 {val_f1 * 100:.2f}%, best val f1 {best_val_f1 * 100:.2f}%', file=log)
        
    
    return train_time, history
                

def test(g, model, args):
    model.load_state_dict(torch.load(os.path.join('./pt', f'{args.dataset}.pt')))
    labels = g.ndata['label']
    test_mask = g.ndata['test_mask']
    test_f1 = evaluate(model, g, labels, test_mask, multilabel=args.multilabel, test=True)
    return test_f1

# def mem_bench(device, g, model, multilabel, args):
#     max_mem = 0
#     torch.cuda.empty_cache()
    
#     subg_iter = SAINTSampler(g, args.num_roots, args.n_layer, args)
    
#     ave_d = []
#     for subg in subg_iter:
#         ave_d.append(subg.in_degrees().sum() / subg.num_nodes())
#     PRINT_LOG(f'ave_d == {math.ceil(np.mean(ave_d))}', file=log)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
#     for j, subg in enumerate(subg_iter):
            
#         subg = subg.to(device)
#         x = subg.ndata['feat'].to(device)
#         y = subg.ndata['label'].to(device)
#         y_pred = model.fullbatch_forward(subg, x)
#         mask = subg.ndata['train_mask'].bool().to(device)
#         if args.multilabel:
#             loss = F.binary_cross_entropy_with_logits(y_pred[mask], y[mask].type_as(y_pred))
#         else:
#             loss = F.cross_entropy(y_pred[mask], y[mask])
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
                        
        
#         peak_usage = torch.cuda.max_memory_allocated(device)
#         max_mem = max(max_mem, peak_usage)
#         torch.cuda.empty_cache()
        
#     PRINT_LOG(f'cuda peak usage: {max_mem / (1024 ** 2):.2f}MB', file=log)
    

    
if __name__ == "__main__":
    args = get_args()
    log = new_log('./log', args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fullbatch_dataset = set(['flickr', 'ogbn-arxiv', 'reddit', 'ppi'])
    minibatch_dataset = set(['amazon', 'ogbn-products', 'yelp'])
    multilabel_dataset = set(['amazon', 'yelp', 'ppi'])
    args.multilabel = args.dataset in multilabel_dataset
    args.fullbatch = args.dataset in fullbatch_dataset
    
    dataset_root = "../dataset"
    if args.dataset == "ogbn-arxiv" or args.dataset == "ogbn-products":
        dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root=dataset_root))
    elif args.dataset == 'aminer':
        dataset = AMinerDataset(dataset_root)
    else:
        dataset = CustomDataset(dataset_root, args.dataset)
    g = dataset[0]
    g = dgl.to_bidirected(g, copy_ndata=True)
    num_classes = dataset.num_classes
    in_feats_size = g.ndata['feat'].shape[1]
    
    tot_test_f1, tot_epoch_train_time = [], []
    
    if args.memory == 1:
        model = GNN(in_feats=in_feats_size,
                    h_feats=args.hid,
                    num_classes=num_classes,
                    n_layer=args.n_layer,
                    dropout=args.dropout,
                    device=device).to(device)
        
        # mem_bench(device, g, model, args.multilabel, args)
    
    else:
    
        for i in range(5):
        
            model = GNN(in_feats=in_feats_size,
                        h_feats=args.hid,
                        num_classes=num_classes,
                        n_layer=args.n_layer,
                        dropout=args.dropout,
                        device=device).to(device)
            
            train_time, history = train(g, model, args, device)
            PRINT_LOG(f'train time: {train_time:.2f}s', file=log)
            test_f1 = test(g, model, args)
            PRINT_LOG(f'test f1: {test_f1 * 100:.2f}%', file=log)
            
            h_time, h_val = [], []
            for temp in history:
                h_time.append(temp[1])
                h_val.append(temp[2])
            print(f'time: {h_time}', file=log)
            print(f'val: {h_val}', file=log)
            
            tot_test_f1.append(test_f1)
            tot_epoch_train_time.append(train_time / args.epoch)
        
        PRINT_LOG(f'test f1: {np.mean(tot_test_f1) * 100:.2f} ± {np.std(tot_test_f1) * 100:.2f}', file=log)
        PRINT_LOG(f'epoch time: {np.mean(tot_epoch_train_time):.2f} ± {np.std(tot_epoch_train_time):.2f}s', file=log)
    
        
    
    

