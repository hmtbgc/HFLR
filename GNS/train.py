from dataset import CustomDataset, AMinerDataset
from models import *
import argparse
import torch
import time
from sklearn.metrics import f1_score
from utils import *
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from dgl import to_block
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from logger import *


def load_subtensor(feats, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = feats[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

class CachedData:
    '''Cache part of the data
    This class caches a small portion of the data and uses the cached data
    to accelerate the data movement to GPU.
    Parameters
    ----------
    node_data : tensor
        The actual node data we want to move to GPU.
    buffer_nodes : tensor
        The node IDs that we like to cache in GPU.
    device : device
        The device where we store cached data.
    '''
    def __init__(self, node_data, buffer_nodes, device):
        num_nodes = node_data.shape[0]
        # Let's construct a vector that stores the location of the cached data.
        # If a node is cached, the corresponding element in the vector stores the location in the cache.
        # If a node is not cached, the element points to the end of the cache.
        self.cached_locs = torch.ones(num_nodes, dtype=torch.int32, device=device) * len(buffer_nodes) # cuda
        self.cached_locs[buffer_nodes] = torch.arange(len(buffer_nodes), dtype=torch.int32, device=device)
        # Let's construct the cache. The last row in the cache doesn't contain valid data.
        self.cache_data = torch.zeros(len(buffer_nodes) + 1, node_data.shape[1], dtype=node_data.dtype, device=device) # cuda
        self.cache_data[:len(buffer_nodes)] = node_data[buffer_nodes].to(device) 
        self.invalid_loc = len(buffer_nodes)
        self.node_data = node_data # cpu

    def __getitem__(self, nids):
        locs = self.cached_locs[nids].long() # cuda
        data = self.cache_data[locs] # cuda
        out_cache_nids = nids[locs == self.invalid_loc] # cuda
        data[locs == self.invalid_loc] = self.node_data[out_cache_nids.cpu()].to(self.cache_data.device) # transfer data in cpu
        return data
    




def train(train_g, val_g, model, args, device):
    torch.cuda.synchronize(device)
    st = time.time()
    
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        fanouts=[args.fan] * args.n_layer
    )
    
    torch.cuda.synchronize(device)
    ed = time.time()
    
    PRINT_LOG(f'preprocess time: {ed - st:.2f}s', file=file)
    
    history = []
    train_time = 0
    best_val_f1 = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    early_stop = 5
    
    train_nid = torch.where(train_g.ndata['train_mask'] == 1)[0]
    
    degree_all_nodes = train_g.in_degrees() + train_g.out_degrees()

    print('get cache sampling probability')
    prob = np.divide(degree_all_nodes, sum(degree_all_nodes))

    
    
    for e in range(args.epoch):
        
        torch.cuda.synchronize(device)
        st = time.time()
        
        
        if e % args.buffer_rs_every == 0:
            if args.buffer_size != 0:
                # initial the buffer
                num_nodes = train_g.num_nodes()
                num_sample_nodes = int(args.buffer_size * num_nodes) 
                buffer_nodes = np.random.choice(num_nodes, num_sample_nodes, replace=False,
                                               p=prob)
                cached_data = CachedData(train_g.ndata['feat'], buffer_nodes, device) # feats: cpu, buffer_nodes: number, cuda
                
            dataloader = dgl.dataloading.DataLoader(
                train_g,
                torch.arange(train_g.num_nodes()),
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                use_uva=True
            )
            
        model.train()

        
        
        for i, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # batch_inputs = blocks[0].srcdata['feat']
            # batch_labels = blocks[-1].dstdata['label']
            
            batch_inputs, batch_labels = load_subtensor(cached_data, train_g.ndata['label'].to(device), seeds, input_nodes, device)
            blocks = [blk.to(device) for blk in blocks]
            pred = model.minibatch_forward(blocks, batch_inputs)
            if (args.multilabel):
                loss = F.binary_cross_entropy_with_logits(pred, batch_labels.type_as(pred), pos_weight=None, reduction="mean")
            else:
                loss = F.cross_entropy(pred, batch_labels, reduction="mean")
    
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        
        torch.cuda.synchronize(device)
        ed = time.time()
        print(f'epoch time == {ed - st:.3f}s')
        train_time += (ed - st)
        
    
        if e > 0 and e % args.every_val == 0:
            model.eval()
            val_f1 = evaluate(model, val_g, val_g.ndata['label'], val_g.ndata['val_mask'], args.multilabel, False)
            history.append((e, round(train_time, 2), round(val_f1, 4)))
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), os.path.join('./pt', f'{args.dataset}.pt'))
            else:
                early_stop -= 1
            PRINT_LOG(f'epoch {e}, val f1 {val_f1 * 100:.2f}%, best val f1 {best_val_f1 * 100:.2f}%', file=file)
    
            
    return train_time, e, history
    
    
def test(test_g, model, args):
    model.load_state_dict(torch.load(os.path.join('./pt', f'{args.dataset}.pt')))
    labels = test_g.ndata['label']
    test_mask = test_g.ndata['test_mask']
    test_f1 = evaluate(model, test_g, labels, test_mask, multilabel=args.multilabel, test=True)
    return test_f1


def mem_bench(device, g, model, multilabel, args):
    max_mem = 0
    torch.cuda.empty_cache()
    
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        fanouts=[args.fan] * args.n_layer
    )

    dataloader = dgl.dataloading.DataLoader(
        g,
        torch.where(g.ndata['train_mask'] == 1)[0],
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    batch_datas = []
    for _, seeds, blocks in tqdm(dataloader):
        batch_datas.append([blocks, seeds])
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    for it, (blocks, seeds) in enumerate(batch_datas):
        blocks = [blk.to(device) for blk in blocks]
        x = blocks[0].srcdata["feat"].to(device)
        y = blocks[-1].dstdata["label"].to(device)
        y_hat = model.minibatch_forward(blocks, x)
        if multilabel:
            loss = F.binary_cross_entropy_with_logits(y_hat, y.type_as(y_hat), reduction="mean")
        else:
            loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        peak_usage = torch.cuda.max_memory_allocated(device)
        max_mem = max(max_mem, peak_usage)
        torch.cuda.empty_cache()
        
    PRINT_LOG(f'cuda peak usage: {max_mem / (1024 ** 2):.2f}MB', file=file)
    
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epoch", type=int, default=50, help="Maximum training epoch")
    parser.add_argument("--every_val", type=int, default=5, help="# epoch for each validation")
    parser.add_argument("--h_feats", type=int, default=128, help="Hidden feature size")
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--fan", type=int, default=5, help="fanout")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout ratio")
    parser.add_argument("--memory", type=int, default=0)
    parser.add_argument('--buffer-size', type=float, default=0.01, help="Buffer size")
    parser.add_argument('--buffer_rs-every', type=int, default=5, help="# epoch for changing buffer")
    args = parser.parse_args()
    
    file = new_log(f'./log', args)    
    fullbatch_eval_dataset = set(['flickr', 'ogbn-arxiv', 'reddit'])
    minibatch_eval_dataset = set(['amazon', 'ogbn-products', 'yelp'])
    multilabel_dataset = set(['amazon', 'yelp', 'ppi'])
    args.multilabel = args.dataset in multilabel_dataset
    args.fullbatch = args.dataset in fullbatch_eval_dataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    in_feats = g.ndata['feat'].shape[1]
    num_classes = dataset.num_classes
    
    tot_test_f1, tot_epoch_train_time = [], []
    
    if args.memory == 1:
        model = GNN(in_feats=in_feats, h_feats=args.h_feats, num_classes=num_classes, n_layer=args.n_layer, dropout=args.dropout, device=device)
        model = model.to(device)
        mem_bench(device, g, model, args.multilabel, args)
    
    else:
    
        for i in range(5):
        
            model = GNN(in_feats=in_feats, h_feats=args.h_feats, num_classes=num_classes, n_layer=args.n_layer, dropout=args.dropout, device=device)
            model = model.to(device)


            train_time, e, history = train(train_g, val_g, model, args, device)
            
            PRINT_LOG(f'train time: {train_time:.2f}s', file=file)
            test_f1 = test(test_g, model, args)
            PRINT_LOG(f'test f1: {test_f1 * 100:.2f}%', file=file)
            
            h_time, h_val = [], []
            for temp in history:
                h_time.append(temp[1])
                h_val.append(temp[2])
            print(f'time: {h_time}', file=file)
            print(f'val: {h_val}', file=file)
            
            tot_test_f1.append(test_f1)
            tot_epoch_train_time.append(train_time / e)
            
        PRINT_LOG(f'test f1: {np.mean(tot_test_f1) * 100:.2f} ± {np.std(tot_test_f1) * 100:.2f}', file=file)
        PRINT_LOG(f'epoch time: {np.mean(tot_epoch_train_time):.2f} ± {np.std(tot_epoch_train_time):.2f}s', file=file)

    
        