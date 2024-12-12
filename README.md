# TransNet

This is a modified pytorch implementation of [_**Augmenting Knowledge Transfer across Graphs**_](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10027706) (ICDM 2022) for **113 Fall Data Mining Final**.

### Environment
1. 創建conda環境
```bash
conda env create -f environment.yml
```

2. 啟動環境
```bash
conda activate dm_transnet # your environment name
```

3. 驗證安裝是否成功
```bash
python -c "import pandas, torch, numpy, torch_geometric, scipy, sklearn, matplotlib, pymetis, structlog, tensorboardX; print('All packages imported successfully!')"
```

4. 我們需要使用CUDA，可以參考 https://medium.com/@yesaouo/cuda-cudnn-tensorflow-gpu-%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8-windows%E9%81%A9%E7%94%A8-%E5%A4%9A%E7%89%88%E6%9C%AC%E5%85%BC%E5%AE%B9-b2688614c506


### Datasets
- The original datasets M2 and A1 are provided in original paper's repository: https://github.com/yuzhenmao/TransNet -> data folder, download "input.zip"
- Some other datasets: https://github.com/TrustAGI-Lab/UDAGCN
- Our experiment datasets download link: 
  - [Cora, CiteSeer, PubMed]
    https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid
  - [CoAuthor CS, CoAuthor Physics]
    https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.datasets.Coauthor.html#torch_geometric.datasets.Coauthor

Please download other datasets from the original papers listed in the paper.

### Implementation Details
_**TransNet**_ is firstly pre-trained on the source dataset for 2000 epochs; then it is fine-tuned on the target dataset for 800 epochs using limited labeled data in each class. We use Adam optimizer with learning rate 3e-3. α in the beta-distribution of trinity-signal mixup is set to 1.0 and the output dimension of MLP in domain unification module is set to 100 by default. Precision is used as the evaluation metric.

### Additional Tools
The `tools` directory contains utility scripts for:
- Converting logs to visualizations
- Other analysis utilities

### Demo Cases

#### Academic Domain Transfer Tasks
[cs+physics]
```bash
python ./src/transnet.py --name='logs_physics_cs_1000_400' --datasets='cs+physics' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01

python ./src/transnet.py --name='logs_physics_cs_2000_800' --datasets='cs+physics' --finetune_epoch=800 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=2000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
```

#### Original Transfer Tasks
[M2+A1]
```bash
python ./src/transnet.py --name='logs_A1_M2_1000_400' --datasets='M2+A1' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01

python ./src/transnet.py --name='logs_A1_M2_2000_800' --datasets='M2+A1' --finetune_epoch=800 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=2000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
```

[comp+photo]
```bash
python ./src/transnet.py --name='logs_photo_comp_1000_400' --datasets='comp+photo' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01

python ./src/transnet.py --name='logs_photo_comp_2000_800' --datasets='comp+photo' --finetune_epoch=800 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=2000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
```

[A2+A1]
```bash
python ./src/transnet.py --name='logs_A1_A2_1000_400' --datasets='A2+A1' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01

python ./src/transnet.py --name='logs_A1_A2_2000_800' --datasets='A2+A1' --finetune_epoch=800 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=2000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
```

[D2+A1]
```bash
python ./src/transnet.py --name='logs_A1_D2_1000_400' --datasets='D2+A1' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01

python ./src/transnet.py --name='logs_A1_D2_2000_800' --datasets='D2+A1' --finetune_epoch=800 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=2000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
```

[A2+D1]
```bash
python ./src/transnet.py --name='logs_D1_A2_1000_400' --datasets='A2+D1' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01

python ./src/transnet.py --name='logs_D1_A2_2000_800' --datasets='A2+D1' --finetune_epoch=800 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=2000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
```

[D2+D1]
```bash
python ./src/transnet.py --name='logs_D1_D2_1000_400' --datasets='D2+D1' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01

python ./src/transnet.py --name='logs_D1_D2_2000_800' --datasets='D2+D1' --finetune_epoch=800 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=2000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
```

[M2+D1]
```bash
python ./src/transnet.py --name='logs_D1_M2_1000_400' --datasets='M2+D1' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01

python ./src/transnet.py --name='logs_D1_M2_2000_800' --datasets='M2+D1' --finetune_epoch=800 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=2000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
```

[pubmed+cora]
```bash
python ./src/transnet.py --name='logs_pubmed_cora_1000_400' --datasets='cora+pubmed' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01

python ./src/transnet.py --name='logs_pubmed_cora_2000_800' --datasets='cora+pubmed' --finetune_epoch=800 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=2000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
```

[pubmed+citeseer]
```bash
python ./src/transnet.py --name='logs_pubmed_citeseer_1000_400' --datasets='citeseer+pubmed' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01

python ./src/transnet.py --name='logs_pubmed_citeseer_2000_800' --datasets='citeseer+pubmed' --finetune_epoch=800 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=2000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
```

For other transfer task examples (M2+D1, A2+A1, A2+D1, D2+A1, D2+D1), please refer to the original repository.

# Reference

Please cite the following paper if you found this library useful in your research:

### Augmenting Knowledge Transfer across Graphs
[Yuzhen Mao](https://scholar.google.com/citations?user=9wKn1A0AAAAJ&hl=en), [Jianhui Sun](https://jsycsjh.github.io/), [Dawei Zhou](https://sites.google.com/view/dawei-zhou/home)\
*IEEE International Conference on Data Mining (ICDM)*, 2022

```bibtex
@inproceedings{mao2022augmenting,
  title={Augmenting Knowledge Transfer across Graphs},
  author={Mao, Yuzhen and Sun, Jianhui and Zhou, Dawei},
  booktitle={2022 IEEE International Conference on Data Mining (ICDM)},
  pages={1101--1106},
  year={2022},
  organization={IEEE}
}
```
