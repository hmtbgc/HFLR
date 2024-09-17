# python new_train.py --dataset flickr --n_layer 3 --fan 5 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --memory 0 --plot

#python new_train.py --dataset ogbn-arxiv --n_layer 2 --fan 5 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --memory 0 --plot

# python new_train.py --dataset reddit --n_layer 3 --fan 5 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --memory 0 --plot

# python new_train.py --dataset ogbn-products --n_layer 3 --fan 5 --h_feats 512 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --memory 0 --plot

# python new_train.py --dataset yelp --n_layer 3 --fan 5 --h_feats 512 --dropout 0.1 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --memory 0 --plot

# python new_train.py --dataset aminer --n_layer 3 --fan 5 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --memory 0 --plot

# python train.py --dataset ogbn-arxiv --n_layer 3 --fan 10 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --memory 0 --alpha 0.01

# python train.py --dataset ogbn-arxiv --n_layer 3 --fan 10 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --memory 0 --alpha 0.05

# python train.py --dataset ogbn-arxiv --n_layer 3 --fan 10 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --memory 0 --alpha 0.1

# python train.py --dataset ogbn-arxiv --n_layer 3 --fan 10 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --memory 0 --alpha 0.2

# python train.py --dataset ogbn-arxiv --n_layer 3 --fan 10 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --memory 0 --alpha 0.3

# python train.py --dataset ogbn-arxiv --n_layer 3 --fan 10 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --memory 0 --alpha 0.4
python train.py --dataset flickr --n_layer 3 --fan 10 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000
python train.py --dataset ogbn-arxiv --n_layer 3 --fan 10 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000
python train.py --dataset reddit --n_layer 3 --fan 10 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000
python train.py --dataset aminer --n_layer 3 --fan 10 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000
python train.py --dataset yelp --n_layer 3 --fan 10 --h_feats 512 --dropout 0.1 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000
python train.py --dataset ogbn-products --n_layer 3 --fan 10 --h_feats 512 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000