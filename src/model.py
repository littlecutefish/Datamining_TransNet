import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from gnns import GNN
import numpy as np
import random
from random import sample
from utils.utils import NodeDistance, mixup_hidden, mixup_hidden_criterion, assign_sudo_label
from multiprocessing import Pool
import networkx as nx
import collections
import pdb


logger = logging.getLogger(__name__)


# GRL Layer
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return inputs                             

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input * ratio

        return grad_input


class TransNet(nn.Module):
    """
    Multi-layer perceptron with adversarial regularizer by domain classification.
    """

    # 初始化模型結構
    # 包含 GNN、域分類器、trinity 分類器等
    def __init__(self, configs):
        super(TransNet, self).__init__()
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_sources = configs["num_sources"]
        self.num_classes = configs["num_classes"]
        self.h_dim = configs['feat_num']
        
        # GNN model 圖神經網路模型
        # 對應算法步驟 1
        # 1-1. 域統一模型
        self.hiddens = GNN( nfeat=self.h_dim,  
                            nhid=configs["hidden_layers"],
                            nclass=2,    # not used
                            ndim=configs["ndim"],
                            gnn_type=configs["type"],
                            bias=True,
                            dropout=configs["dropout"])
        self.domain_disc = torch.nn.Sequential( 
            nn.Linear(configs["ndim"], configs["ndim"]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(configs["ndim"], 2),
        )
        self.domain_disc_linear = torch.nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.h_dim, 2),
        )

        # Parameter of the first dimension reduction layer.
        self.dimRedu = nn.ModuleList([torch.nn.Sequential(
            nn.Dropout(p=0.7), 
            nn.Linear(ndim, self.h_dim)) 
            for ndim in configs["input_dim"]])
        
        # Parameter of the final softmax classification layer.
        # 1-2. trinity-signal 分類器 g(.)
        self.triplet_embedding = nn.ModuleList([torch.nn.Sequential(
                                            nn.Linear(2*configs["ndim"], configs["ndim"]),
                                            ) for _ in configs["num_classes"]])
        self.classifier = nn.ModuleList([torch.nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(configs["ndim"], 2*nclass)) for nclass in configs["num_classes"]])

        # 1-3. 下游任務分類器 h(.)
        self.gnn_classifier = GNN(nfeat=configs["ndim"], 
                                nhid=[configs["ndim"]], 
                                nclass=2, 
                                ndim=configs["num_classes"][-1],
                                gnn_type=configs["type"], 
                                bias=True, 
                                dropout=0.5, 
                                batch_norm=False)
        self.domains = nn.ModuleList([self.domain_disc for _ in range(self.num_sources)])
        self.domains_linear = nn.ModuleList([self.domain_disc_linear for _ in range(self.num_sources)])
        # Gradient reversal layer.
        self.grls = [GradientReversalLayer.apply for _ in range(self.num_sources)]
        self.t_class = configs["num_classes"][-1]
        self.t_dim = configs["ndim"]

    # 對應步驟2-6的 pre-train 階段
    # 計算域不變表示
    # 生成 trinity signals
    # M1
    def forward(self, sinputs, tinputs, sadj, tadj, rate):
        global ratio
        ratio = rate

        # 步驟3: 計算域不變表示
        sh_relu = []
        sh_linear = []
        th_relu = tinputs.clone()
        for i in range(self.num_sources):
            sh_relu.append(sinputs[i].clone())

        # 通過feature encoder (MLP)和GNN獲取域不變特徵
        for i in range(self.num_sources):
            sh_linear.append(self.dimRedu[i](sh_relu[i]))  # MLP
            sh_relu[i] = F.relu(self.hiddens(sh_linear[i], sadj[i]))  # GNN
        th_linear = self.dimRedu[-1](th_relu)
        th_relu = F.relu(self.hiddens(th_linear, tadj))

        # 步驟4: 生成trinity signals (在訓練循環中處理)
        sdomains, tdomains, sdomains_linear, tdomains_linear = [], [], [], []
        for i in range(self.num_sources):
            sdomains.append(F.log_softmax(self.domains[i](self.grls[i](sh_relu[i])), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.grls[i](th_relu)), dim=1))
            sdomains_linear.append(F.log_softmax(self.domains_linear[i](self.grls[i](sh_linear[i])), dim=1))
            tdomains_linear.append(F.log_softmax(self.domains_linear[i](self.grls[i](th_linear)), dim=1))

        return sh_relu, th_relu, sh_linear, th_linear, sdomains, tdomains, sdomains_linear, tdomains_linear

    # 對應步驟7-9的 fine-tune 階段
    # 凍結特定層進行微調
    def finetune(self):
        # 凍結特定層進行微調
        for param in self.hiddens.parameters():
            param.requires_grad = False
        for param in self.dimRedu[-1].parameters():
            param.requires_grad = False

    def finetune_inv(self):
        for param in self.hiddens.parameters():
            param.requires_grad = True
        for param in self.dimRedu[-1].parameters():
            param.requires_grad = True

    # 用於預測階段
    # 對目標域數據進行推理
    def inference(self, tinput, adj, index=-1, pseudo=False):
        h_relu = tinput.clone()
        h_linear = self.dimRedu[index](h_relu)
        h_relu = F.relu(self.hiddens(h_linear, adj))
        if pseudo is True:
            index = 0
        if index == -1:
            logprobs = F.log_softmax(self.gnn_classifier(h_relu, adj), dim=1)
        else:
            logprobs = F.log_softmax(self.classifier[index](self.triplet_embedding[index](torch.cat((h_relu, h_relu), 1)))[:, :self.num_classes[index]], dim=1)
        return logprobs, h_relu, h_linear


# ======================================================
class GeneralSignal(object):
    def __init__(self, graph, adj, labels, features, nhid, device, dataset, args, seed, model, index, _lambda, dicts=None):
        self.G = graph.G
        self.model = model
        self.node_num = len(self.G)
        self.name = graph.name
        self.adj = adj.clone()
        self.adj = self.adj.to_dense().cpu().numpy()
        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device
        self.labels = labels
        self.nlabel = self.labels.max()+1
        self.index = index
        self.idx_train = np.sort(np.array(dataset.train_indices))
        self.idx_test = np.sort(np.array(dataset.test_indices))
        self.idx_vali = np.sort(np.array(dataset.val_indices))
        self.idx_finetune = np.sort(np.array(dataset.finetune_indices))
        self._lambda = _lambda
        self.alpha = args.alpha
        self.mixup = args.no_mixup
        self.dicts = dicts
        
        random.seed(seed)
        np.random.seed(seed)

        self.all = np.arange(self.adj.shape[0])
        
        # 移除所有距離相關計算,改為簡單的隨機採樣準備
        self.base_samples = min(500, len(self.idx_train) // 10)
        self.sample_num = [self.base_samples * (i+1) for i in range(6)]
        self.train_sample_num = [self.base_samples * (i+1) // 2 for i in range(6)]
        
        print("\n=== Dataset Information ===")
        print(f"Training set size: {len(self.idx_train)}")
        print(f"Total nodes: {self.node_num}")
        
        print("\n=== Sampling Configuration ===")
        print(f"Base samples: {self.base_samples}")
        print(f"Sample numbers: {self.sample_num}")
        print(f"Train sample numbers: {self.train_sample_num}")
        print("============================\n")

    def random_sample(self, num_samples, use_train=False):
        """
        隨機採樣方法
        """
        if use_train:
            available_indices = np.arange(len(self.idx_train))
            max_samples = len(self.idx_train)
        else:
            available_indices = np.arange(self.node_num)
            max_samples = self.node_num
            
        safe_num_samples = min(num_samples, max_samples)
        
        idx1 = np.random.choice(available_indices, safe_num_samples, replace=True)
        idx2 = np.random.choice(available_indices, safe_num_samples, replace=True)
        
        return idx1, idx2

    def get_dicts(self, embeddings, idx_assis, labels, relax=False):
        """
        為 few-shot 學習生成字典映射
        """
        
        permu = np.random.permutation(idx_assis.shape[0])
        few_shot_pair = torch.cat((embeddings[idx_assis], embeddings[idx_assis][permu]), 1)
        few_shot_embedding = self.model.triplet_embedding[0](few_shot_pair)
        few_shot_output = self.model.classifier[0](few_shot_embedding)
        sudo_label_num = few_shot_output.shape[-1] // 2
        
        few_shot_pred_label0 = F.log_softmax(few_shot_output[:, :sudo_label_num], dim=1)
        few_shot_pred_label1 = F.log_softmax(few_shot_output[:, sudo_label_num:], dim=1)
        few_shot_pred_label = torch.cat((few_shot_pred_label0, few_shot_pred_label1), 0)
        few_shot_label = torch.cat((labels, labels[permu]), 0)
        
        sudo_labels, self.dicts = assign_sudo_label(few_shot_pred_label, few_shot_label, self.device, relax=relax)
        
        return sudo_labels

    def make_loss(self, embeddings, dicts=None, task='train'):
        if self.index == -1:
            num_samples = min(sum(self.sample_num), self.node_num)
            
            node_pairs = self.random_sample(num_samples)
            embeddings0 = embeddings[node_pairs[0]]
            embeddings1 = embeddings[node_pairs[1]]

            permu = np.random.permutation(self.idx_finetune.shape[0])
            few_shot_pair = torch.cat((embeddings[self.idx_finetune], embeddings[self.idx_finetune][permu]), 1)
            few_shot_embedding = self.model.triplet_embedding[0](few_shot_pair)
            few_shot_output = self.model.classifier[0](few_shot_embedding)
            sudo_label_num = few_shot_output.shape[-1] // 2
            
            few_shot_pred_label0 = F.log_softmax(few_shot_output[:, :sudo_label_num], dim=1)
            few_shot_pred_label1 = F.log_softmax(few_shot_output[:, sudo_label_num:], dim=1)
            few_shot_pred_label = torch.cat((few_shot_pred_label0, few_shot_pred_label1), 0)
            few_shot_label = torch.cat((self.labels[self.idx_finetune], self.labels[self.idx_finetune][permu]), 0)
            
            if few_shot_label.max() >= sudo_label_num:
                few_shot_label = torch.clamp(few_shot_label, 0, sudo_label_num - 1)
            
            sudo_labels, _ = assign_sudo_label(few_shot_pred_label, few_shot_label, self.device, dicts=dicts)
        
            label0_loss, label1_loss = 0.0, 0.0 
            
            # Mixup 只考慮標籤
            if self.mixup:
                iter_num = 1
                for _ in range(iter_num):
                    x_mix, permuted_idx, lam = mixup_hidden(few_shot_embedding, self.alpha)
                    o_mix = self.model.classifier[0](x_mix)
                    p_mix0 = o_mix[:, :sudo_label_num]
                    p_mix1 = o_mix[:, sudo_label_num:]
                    
                    labels0 = sudo_labels[:self.idx_finetune.shape[0]].clone()
                    labels1 = sudo_labels[self.idx_finetune.shape[0]:].clone()
                    labels0 = torch.clamp(labels0, 0, sudo_label_num - 1)
                    labels1 = torch.clamp(labels1, 0, sudo_label_num - 1)
                    
                    label0_loss += mixup_hidden_criterion(p_mix0, labels0, 
                                                        labels0[permuted_idx], lam)
                    label1_loss += mixup_hidden_criterion(p_mix1, labels1, 
                                                        labels1[permuted_idx], lam)
                label0_loss, label1_loss = label0_loss/iter_num, label1_loss/iter_num
            else:
                label0_loss = F.nll_loss(few_shot_pred_label0, 
                                       torch.clamp(sudo_labels[:self.idx_finetune.shape[0]], 0, sudo_label_num - 1))
                label1_loss = F.nll_loss(few_shot_pred_label1, 
                                       torch.clamp(sudo_labels[self.idx_finetune.shape[0]:], 0, sudo_label_num - 1))
        else:
            if task == 'train':
                num_samples = min(sum(self.train_sample_num), len(self.idx_train))
                node_pairs = self.random_sample(num_samples, use_train=True)
                embeddings0 = embeddings[self.idx_train[node_pairs[0]]]
                embeddings1 = embeddings[self.idx_train[node_pairs[1]]]
            elif task == 'vali':
                node_pairs = [self.idx_vali, self.idx_vali]
                embeddings0 = embeddings[node_pairs[0]]
                embeddings1 = embeddings[node_pairs[1]]

            embedding_pair = torch.cat((embeddings0, embeddings1), 1)
            hidden_embedding = self.model.triplet_embedding[self.index](embedding_pair)
            output = self.model.classifier[self.index](hidden_embedding)

            if task == 'train':
                task_index = self.idx_train
            elif task == 'vali':
                task_index = self.all

            pred_label0 = output[:, :self.nlabel]
            pred_label1 = output[:, self.nlabel:]
            pred_label0 = F.log_softmax(pred_label0, dim=1)
            pred_label1 = F.log_softmax(pred_label1, dim=1)
            
            labels0 = torch.clamp(self.labels[task_index[node_pairs[0]]], 0, self.nlabel - 1)
            labels1 = torch.clamp(self.labels[task_index[node_pairs[1]]], 0, self.nlabel - 1)
            
            label0_loss = F.nll_loss(pred_label0, labels0)
            label1_loss = F.nll_loss(pred_label1, labels1)

            if task == 'train' and self.mixup:
                x_mix, permuted_idx, lam = mixup_hidden(hidden_embedding, self.alpha)
                o_mix = self.model.classifier[self.index](x_mix)
                p_mix0 = o_mix[:, :self.nlabel]
                p_mix1 = o_mix[:, self.nlabel:]
                
                label0_loss = mixup_hidden_criterion(p_mix0, labels0, labels0[permuted_idx], lam)
                label1_loss = mixup_hidden_criterion(p_mix1, labels1, labels1[permuted_idx], lam)
        
        return (label0_loss + label1_loss) / 2
    