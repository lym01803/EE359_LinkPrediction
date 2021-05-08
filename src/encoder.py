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
        self.Embeddings = torch.zeros(self.G.n, self.dim).to("cuda:0")
        self.Embeddings.requires_grad = False
        nn.init.normal_(self.Embeddings, mean=0.0, std=1.0/(self.dim**0.5))
        self.UnigramTable = list()
        print("InitUnigramTable...")
        self.initUnigramTable()
        self.ridx = 0
        # self.optimizer = torch.optim.Adam([self.Embeddings], lr=0.001)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda e: 0.5 if ((e + 1) % 10 == 0) else 1.0)

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
        # iter_loss = 0
        iter_gradient_norm = 0
        for iter in tqdm(range(max_iter)):
            #loss_list = []
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
                # self.optimizer.zero_grad()
                # loss = torch.Tensor([0.0]).to("cuda:0")
                '''
                for v in walk[1:]:
                    # loss -= torch.log(torch.sigmoid(torch.matmul(self.Embeddings[u], self.Embeddings[v])))
                    p = torch.sigmoid(torch.matmul(self.Embeddings[u], self.Embeddings[v]))
                    self.Embeddings[u] += lr * (1.0 - p) * self.Embeddings[v]
                    self.Embeddings[v] += lr * (1.0 - p) * self.Embeddings[u]
                '''
                embedu = self.Embeddings[u]

                idx_p = torch.tensor(walk[1:], dtype=torch.int64).to("cuda:0")
                embedvs_p = torch.index_select(self.Embeddings, 0, idx_p)
                p_p = 1.0 - torch.sigmoid(torch.matmul(embedvs_p, embedu))

                # self.Embeddings[u] += lr * torch.matmul(p_, embedvs)
                # self.Embeddings.index_add_(0, idx, lr * (p_.view(-1, 1) * embedu))
                '''
                for k in range(negative):
                    ridx = random.randint(0, unisize)
                    v = self.UnigramTable[ridx]
                    if v in self.G.A_set[u]:
                        continue
                    p = torch.sigmoid(-torch.matmul(self.Embeddings[u], self.Embeddings[v]))
                    self.Embeddings[u] -= lr * (1.0 - p) * self.Embeddings[v]
                    self.Embeddings[v] -= lr * (1.0 - p) * self.Embeddings[u]
                    # loss -= torch.log(torch.sigmoid(- torch.matmul(self.Embeddings[u], self.Embeddings[v])))
                '''
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
                        
                idx_n = torch.tensor(neighbors, dtype=torch.int64).to("cuda:0")
                embedvs_n = torch.index_select(self.Embeddings, 0, idx_n)
                # embedu = self.Embeddings[u]
                p_n = 1.0 - torch.sigmoid(-torch.matmul(embedvs_n, embedu))

                dEu = lr * (torch.matmul(p_p, embedvs_p) - torch.matmul(p_n, embedvs_n)) - weight_decay * embedu
                self.Embeddings[u] += dEu
                # self.Embeddings[u] += lr * torch.matmul(p_p, embedvs_p)
                # self.Embeddings[u] -= lr * torch.matmul(p_n, embedvs_n)
                self.Embeddings.index_add_(0, idx_p, lr * (p_p.view(-1, 1) * embedu) - weight_decay * embedvs_p)
                self.Embeddings.index_add_(0, idx_n, -lr * (p_n.view(-1, 1) * embedu) - weight_decay * embedvs_n)
                    
                if (iter + 1) % 10 == 0:
                    # iter_loss += torch.sum(torch.log(1.0 - p_p)).item()
                    # iter_loss += torch.sum(torch.log(1.0 - p_n)).item()
                    iter_gradient_norm += (torch.norm(dEu) / lr).item()
                # loss.backward()
                # self.optimizer.step()
                # loss_list.append(loss.item())
            # print(iter, sum(loss_list) / len(loss_list))
            # self.scheduler.step()
            if (iter + 1) % 25 == 0:
                lr *= 0.5
            if (iter + 1) % 10 == 0:
                # print("iter : {} ; average_loss : {}".format(iter + 1, iter_loss / iter_length))
                # iter_loss = 0
                print('iter : {} ; average_gradient_norm : {}'.format(iter+1, 
                    iter_gradient_norm / iter_length))
                iter_gradient_norm = 0            

    def save_embeddings(self, pth):
        torch.save(self.Embeddings, pth)
    