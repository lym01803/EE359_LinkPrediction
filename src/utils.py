import numpy as np
import random

import encoder
import torch

class Graph:
    def __init__(self, E, n=None):
        self.E = E # a numpy array, shape (2 * |E|)
        if n is None:
            n = E.max() + 1
        self.n = n
        self.A, self.A_set = self.get_adj_table()
        self.Walks = list()

    def get_adj_table(self):
        A = [[] for i in range(self.n)]
        edges = set()
        for e in range(self.E.shape[1]):
            u, v = self.E[0][e], self.E[1][e]
            if not (u, v) in edges:
                edges.add((u, v))
                edges.add((v, u))
                A[u].append(v)
                A[v].append(u)
        A_set = [set(Nu+[i]) for i, Nu in enumerate(A)]
        return A, A_set

    def calc_alpha_t_x(self, t, x, invp, invq):
        if x == t: # dtx = 0
            return invp
        if x in self.A_set[t]: # dtx = 1
            return 1.0
        return invq # dtx = 2
    
    def perform_walk(self, u, p=1.0, q=1.0, length=20):
        invp = 1.0 / p
        invq = 1.0 / q
        walk = [u]
        t = u
        if len(self.A[u]) == 0:
            return
        while len(walk) < length:
            Nu = self.A[u]
            W = list()
            for ni in Nu:
                W.append(self.calc_alpha_t_x(t, ni, invp, invq))
            t = u
            idx = np.argmax(np.random.multinomial(1, np.array(W) / sum(W)))
            u = Nu[idx]
            walk.append(u)
        self.Walks.append(walk)

    def clean_walk(self, left=0):
        if left == 0:
            self.Walks = list()
            return 
        if left > len(self.Walks):
            left = len(self.Walks)
        self.Walks = self.Walks[-left: ]


def Read_Graph(pth, split=False, validsize=3000):
    E = np.loadtxt(pth, dtype=np.int, skiprows=1, delimiter=',')
    if not split:
        return E.T
    E_ = E.tolist()
    random.shuffle(E_)
    ETrain = E_[validsize:]
    EValid = E_[:validsize]
    return (np.array(ETrain, dtype=np.int)).T, (np.array(EValid, dtype=np.int)).T

def Generate_Negative(G, num):
    count = 0
    E = [[], []]
    n = G.n
    while count < num:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if not v in G.A_set[u]:
            count += 1
            E[0].append(u)
            E[1].append(v)
    return np.array(E)

if __name__ == '__main__':
    E = np.array([
        [0, 0, 0, 0, 0, 5, 5, 6],
        [1, 2, 3, 4, 5, 6, 7, 7]
    ], dtype=np.int)
    G = Graph(E)
    for i in range(G.n):
        for r in range(5):
            G.perform_walk(i, 2, 0.5, 20)
    for w in G.Walks:
        print(w)

    Model = encoder.Node2Vec(G, dim=128)
    Model.training()
    for i in range(G.n):
        print(Model.Embeddings[i])

    for i in range(100):
        u, v = input().split()
        u = int(u)
        v = int(v)
        eu = Model.Embeddings[u]
        ev = Model.Embeddings[v]
        print(torch.matmul(eu, ev))
    
