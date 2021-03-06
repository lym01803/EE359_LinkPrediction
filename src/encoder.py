import random
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from tqdm import tqdm

class Node2Vec:
    def __init__(self, G, dim=128):
        self.G = G
        self.dim = dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Embeddings = torch.zeros(self.G.n, self.dim).to(self.device)
        self.Embeddings.requires_grad = False
        nn.init.normal_(self.Embeddings, mean=0.0, std=1.0/(self.dim**0.5))
        self.UnigramTable = list()
        print("InitUnigramTable...")
        self.initUnigramTable()
        self.ridx = 0

    def initUnigramTable(self, size=10000000):
        count = [len(self.G.A[i])+1 for i in range(self.G.n)]
        count = np.array(count) ** 0.75
        total_pow = np.sum(count)
        i = 0
        prefix_sum = count[i] / total_pow
        for idx in tqdm(range(size)):
            self.UnigramTable.append(i)
            if idx > prefix_sum * size:
                i += 1
                if i >= self.G.n:
                    i = self.G.n - 1
                prefix_sum += count[i] / total_pow
        print('shuffling...')
        random.shuffle(self.UnigramTable)
    
    def training(self, max_iter=100, negative=100, rewalk=True, weight_decay=1e-4):
        unisize = len(self.UnigramTable)
        lr = 0.01
        iter_gradient_norm = 0
        for iter in tqdm(range(max_iter)):
            iter_length = 0
            if rewalk:
                print('Random walking...')
                self.G.clean_walk()
                for i in tqdm(range(self.G.n)):
                    self.G.perform_walk(i, 2, 0.5, 20)
                iter_length = len(self.G.Walks)
            else:
                iter_length = self.G.n
            random.shuffle(self.G.Walks)

            for walk in tqdm(self.G.Walks[:iter_length]):
                u = walk[0]
                embedu = self.Embeddings[u]

                idx_p = torch.tensor(walk[1:], dtype=torch.int64).to(self.device)
                embedvs_p = torch.index_select(self.Embeddings, 0, idx_p)
                p_p = 1.0 - torch.sigmoid(torch.matmul(embedvs_p, embedu))

                # negative sample
                neighbors = list()
                count = 0
                while count < negative:
                    v = self.UnigramTable[self.ridx]
                    if not v in self.G.A_set[u]:
                        neighbors.append(v)
                        count += 1
                    self.ridx += 1
                    if self.ridx >= len(self.UnigramTable):
                        self.ridx = 0
                        
                idx_n = torch.tensor(neighbors, dtype=torch.int64).to(self.device)
                embedvs_n = torch.index_select(self.Embeddings, 0, idx_n)
                p_n = 1.0 - torch.sigmoid(-torch.matmul(embedvs_n, embedu))

                # gradient descent
                dEu = lr * (torch.matmul(p_p, embedvs_p) - torch.matmul(p_n, embedvs_n)) - weight_decay * embedu
                self.Embeddings[u] += dEu
                self.Embeddings.index_add_(0, idx_p, lr * (p_p.view(-1, 1) * (embedu - dEu)) - weight_decay * embedvs_p)
                self.Embeddings.index_add_(0, idx_n, -lr * (p_n.view(-1, 1) * (embedu - dEu)) - weight_decay * embedvs_n)
                    
                if (iter + 1) % 10 == 0:
                    iter_gradient_norm += (torch.norm(dEu) / lr).item()
                    
            if (iter + 1) % 25 == 0:
                lr *= 0.5
            if (iter + 1) % 10 == 0:
                print('iter : {} ; average_gradient_norm : {}'.format(iter+1, 
                    iter_gradient_norm / iter_length))
                iter_gradient_norm = 0            

    def save_embeddings(self, pth):
        torch.save(self.Embeddings, pth)
    