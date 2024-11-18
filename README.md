# TransNet

This is a modified pytorch implementation of [_**Augmenting Knowledge Transfer across Graphs**_](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10027706) (ICDM 2022).

### Environment
#### ❯ python --version
Python 3.8.20

#### ❯ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:41:10_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.8, V11.8.89

### Datasets
The original datasets M2 and A1 are provided in this repository. Additional datasets (physics and cs) have been added for academic domain transfer learning experiments. Please download other datasets from the original papers listed in the paper.

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
# Using GCN
python ./src/transnet.py --name='logs_physics_cs_1000_400' --datasets='cs+physics' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01

# Using GAT
python ./src/transnet.py --name='logs_physics_cs_1000_400_GAT' --datasets='cs+physics' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gat' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
```

#### Original Transfer Tasks
[M2+A1]
```bash
python ./src/transnet.py --name='logs_M2+A1_1000_400_4' --datasets='M2+A1' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=1000  --batch_size=4   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01

python ./src/transnet.py --name='logs_A1_M2_1000_400_GAT' --datasets='M2+A1' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gat' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
```

[comp+photo]
```bash
python ./src/transnet.py --name='logs_photo_comp_1000_400' --datasets='comp+photo' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gcn' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01

python ./src/transnet.py --name='logs_photo_comp_1000_400_GAT' --datasets='comp+photo' --finetune_epoch=400 --mu=1e-2 --seed=100 --gnn='gat' --few_shot=5  --epoch=1000  --batch_size=-1   --finetune_lr=0.01  --pre_finetune=200 --ratio=0.7 --disc='3' --_lambda=0.02  --_lambda=0.05 --_alpha=0.01 --_alpha=0.01
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