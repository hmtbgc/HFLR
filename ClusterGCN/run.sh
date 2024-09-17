python train.py --dataset flickr --lr 0.005 --wd 0.0 --epoch 100 --every_val 5 --h_feats 256 --n_layer 3 --each_part 3000 --batch_size 2 --dropout 0.5
python train.py --dataset ogbn-arxiv --lr 0.005 --wd 0.0 --epoch 100 --every_val 5 --h_feats 256 --n_layer 3 --each_part 3000 --batch_size 2 --dropout 0.5
python train.py --dataset reddit --lr 0.005 --wd 0.0 --epoch 100 --every_val 5 --h_feats 256 --n_layer 3 --each_part 3000 --batch_size 2 --dropout 0.5
python train.py --dataset aminer --lr 0.005 --wd 0.0 --epoch 100 --every_val 5 --h_feats 256 --n_layer 3 --each_part 3000 --batch_size 2 --dropout 0.5
python train.py --dataset yelp --lr 0.005 --wd 0.0 --epoch 100 --every_val 5 --h_feats 512 --n_layer 3 --each_part 3000 --batch_size 2 --dropout 0.1
python train.py --dataset ogbn-products --lr 0.005 --wd 0.0 --epoch 100 --every_val 5 --h_feats 512 --n_layer 3 --each_part 3000 --batch_size 2 --dropout 0.5

