# HFLR
code for HFLR

### How to Run
Download dataset from [Google Drive link](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) or [BaiduYun link(code:f1ao)](https://pan.baidu.com/share/init?surl=SOb0SiSAXavwAcNqkttwcg) and put it at correct place:

```
/home/user/tot_code
│   
└───GraphSAGE
|   
└───GraphSAINT
| 
└───dataset/
|   |   flickr/
|   |   ogbn-arxiv/
|   |   ...
```

Only four files are needed: adj_full.npz, class_map.json, feats.npy and role.json. These public datasets are collected by GraphSAINT and are irrelevant to this paper.

For graph AMiner, you can download it from [Google Drive link](https://drive.google.com/file/d/1yG5BP0GJKoB2Q07Uqd1DuC2tMf4EZo4u/view) or [BaiduYun link(code:l0pe)](https://pan.baidu.com/share/init?surl=QWsioe2hPTFWyoL3aF6jlQ).

If you want to run certain algorithm(e.g. HFLR):
```shell
cd HFLR
bash run.sh
```

Results will be saved at log directory.