import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from models import GNN
from utils import *
import dgl
from dataset import *
import os
from logger import *
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
import math
import argparse

def train(train_g, val_g, model, args, device):
    torch.cuda.synchronize(device)
    st = time.time()
    
    partition_id = dgl.metis_partition_assignment(train_g, args.num_part)
    partitions = []
    for id in range(args.num_part):
        partitions.append(torch.where(partition_id == id)[0])
        
    
    torch.cuda.synchronize(device)
    ed = time.time()
    
    PRINT_LOG(f'preprocess time: {ed - st:.2f}s', file=log)
    
    history = []
    train_time = 0
    best_val_f1 = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    
    for epoch in range(args.epoch):
        
        torch.cuda.synchronize(device)
        st = time.time()
        model.train()
        
        index = torch.randperm(args.num_part)
        for i in range(0, args.num_part, args.batch_size):
            id = index[i : i + args.batch_size]
            node_id = torch.concat([partitions[j] for j in id], dim=0)
            node_id = torch.unique(node_id)
            sg = train_g.subgraph(node_id).to(device)
            x = sg.ndata['feat'].to(device)
            y = sg.ndata['label'].to(device)
            y_pred = model.fullbatch_forward(sg, x)
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
            val_f1 = evaluate(model, val_g, val_g.ndata['label'], val_g.ndata['val_mask'], args.multilabel, False)
            history.append((epoch, round(train_time, 2), round(val_f1, 4)))
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), os.path.join('./pt', f'{args.dataset}.pt'))
            PRINT_LOG(f'epoch {epoch}, val f1 {val_f1 * 100:.2f}%, best val f1 {best_val_f1 * 100:.2f}%', file=log)
        
    
    return train_time, history
            
            
        

def test(test_g, model, args):
    model.load_state_dict(torch.load(os.path.join('./pt', f'{args.dataset}.pt')))
    labels = test_g.ndata['label']
    test_mask = test_g.ndata['test_mask']
    test_f1 = evaluate(model, test_g, labels, test_mask, multilabel=args.multilabel, test=True)
    return test_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epoch", type=int, default=50, help="Maximum training epoch")
    parser.add_argument("--every_val", type=int, default=5, help="# epoch for each validation")
    parser.add_argument("--h_feats", type=int, default=128, help="Hidden feature size")
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--each_part", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout ratio")
    parser.add_argument("--memory", type=int, default=0)
    args = parser.parse_args()
    log = new_log('./log', args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    fullbatch_eval_dataset = set(['flickr', 'ogbn-arxiv', 'reddit', 'ppi'])
    minibatch_eval_dataset = set(['amazon', 'ogbn-products', 'yelp'])
    multilabel_dataset = set(['amazon', 'yelp', 'ppi'])
    args.multilabel = args.dataset in multilabel_dataset
    args.fullbatch = args.dataset in fullbatch_eval_dataset
    
    dataset_root = "../dataset"
    if args.dataset == "ogbn-arxiv" or args.dataset == "ogbn-products":
        dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root=dataset_root))
    elif args.dataset == 'aminer':
        dataset = AMinerDataset(dataset_root)
    else:
        dataset = CustomDataset(dataset_root, args.dataset)
    g = dataset[0]
    g = dgl.to_bidirected(g, copy_ndata=True)
    
    train_g = g.subgraph(g.ndata['train_mask'].bool())
    val_g = g.subgraph(g.ndata['train_mask'].bool() | g.ndata['val_mask'].bool())
    test_g = g
    
    num_part = math.ceil(train_g.num_nodes() / args.each_part) * 2
    args.num_part = num_part
    
    in_feats = g.ndata['feat'].shape[1]
    num_classes = dataset.num_classes
    
    tot_test_f1, tot_epoch_train_time = [], []
    
    for i in range(5):
        
        model = GNN(in_feats=in_feats, h_feats=args.h_feats, num_classes=num_classes, n_layer=args.n_layer, dropout=args.dropout, device=device)
        model = model.to(device)

        train_time, history = train(train_g, val_g, model, args, device)
        
        PRINT_LOG(f'train time: {train_time:.2f}s', file=log)
        test_f1 = test(test_g, model, args)
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

    
    