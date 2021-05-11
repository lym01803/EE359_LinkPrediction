import numpy as np
import torch
import encoder
import utils
from tqdm import tqdm
import random
import argparse

def Train(ET):
    global args
    G = utils.Graph(ET)
    print('Random Walking...')
    walk_num = 10
    walk_length = 25
    nega_ratio = 10
    for i in tqdm(range(G.n)):
        for r in range(walk_num):
            # G.perform_walk(i, random.lognormvariate(0, 0.7), random.lognormvariate(0, 0.7), walk_length) # 0.7 \approx log 2
            G.perform_walk(i, p=2.0, q=1.0, length=walk_length)

    Model = encoder.Node2Vec(G, dim=128)
    print('Learning Embedding...')
    Model.training(max_iter=args.max_iteration, negative=walk_length*nega_ratio, rewalk=False)
    Model.save_embeddings(args.save_path)
    return Model

def Valid(EV, E, model=None):
    if model is None:
        pass
    Embeddings = model.Embeddings
    PositiveSample = EV
    NegativeSample = utils.Generate_Negative(utils.Graph(E), num=PositiveSample.shape[1])
    Euv = torch.matmul(Embeddings, Embeddings.T)
    Euu = torch.sum(Embeddings * Embeddings, dim=1)
    Posi_score = []
    Nega_score = []
    for i in range(PositiveSample.shape[1]):
        u, v = PositiveSample[0][i], PositiveSample[1][i]
        Posi_score.append((Euv[u][v] / torch.sqrt(Euu[u]) / torch.sqrt(Euu[v])).item())
    for i in range(NegativeSample.shape[1]):
        u, v = NegativeSample[0][i], NegativeSample[1][i]
        Nega_score.append((Euv[u][v] / torch.sqrt(Euu[u]) / torch.sqrt(Euu[v])).item())
    count = 0
    for p in Posi_score:
        for n in Nega_score:
            if p >= n:
                count += 1
    print(count, len(Posi_score), len(Nega_score))
    with open('../data/posi.csv', 'w', encoding='utf8') as f:
        for i in range(PositiveSample.shape[1]):
            f.write('{} {} : {}\n'.format(PositiveSample[0][i], PositiveSample[1][i], Posi_score[i]))
    with open('../data/nega.csv', 'w', encoding='utf8') as f:
        for i in range(NegativeSample.shape[1]):
            f.write('{} {} : {}\n'.format(NegativeSample[0][i], NegativeSample[1][i], Nega_score[i]))
    return 1.0 * count / len(Posi_score) / len(Nega_score)

def Test(Embeddings, ETest):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Embeddings = Embeddings.to(device)
    Euv = torch.matmul(Embeddings, Embeddings.T)
    Euu = torch.sum(Embeddings * Embeddings, dim=1)
    score = []
    for i in range(ETest.shape[1]):
        u, v = ETest[0][i], ETest[1][i]
        score.append((Euv[u][v] / torch.sqrt(Euu[u]) / torch.sqrt(Euu[v])).item())
    return score

parser = argparse.ArgumentParser()
parser.add_argument('--valid', action='store_true')
parser.add_argument('--train_from_scratch', action='store_true')
parser.add_argument('--embedding_path', type=str, default='../data/embedding_128')
parser.add_argument('--result_path', type=str, default='../submission.csv')
parser.add_argument('--save_path', type=str, default='../data/embedding_128_new')
parser.add_argument('--max_iteration', type=int, default=100)

global args
args = parser.parse_args()

if args.valid:
    ET, EV = utils.Read_Graph('../data/course3_edge.csv', split=True, validsize=6000)
    model = Train(ET)
    Res = Valid(EV, np.hstack((ET, EV)), model)
    print("AUC: ", Res)
else:
    Embeddings = None
    if args.train_from_scratch:
        E = utils.Read_Graph('../data/course3_edge.csv', split=False)
        model = Train(E)
        Embeddings = model.Embeddings
    else:
        Embeddings = torch.load(args.embedding_path)
    ETest = utils.Read_Test('../data/course3_test.csv')
    score = Test(Embeddings, ETest)
    with open(args.result_path, 'w', encoding='utf8') as f:
        f.write('id,label\n')
        for i, s in enumerate(score):
            f.write('{},{:.4f}\n'.format(i, 0.5 * (s + 1.0)))
