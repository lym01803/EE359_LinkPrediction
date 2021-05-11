# EE359 Lab3 Link Prediction With Node Embedding Method (DeepWalk and Node2Vec)

## Environment
- python 3.7.6 (python 3 is OK)
- torch 1.6.0 with gpu
- CUDA 10.2

Claim: I do not use any advanced API. I only use some basic functions in ``torch`` such as ``torch.matmul``. I used to use autograd, but I gave up due to efficiency.

## For Reproducing
Get the link prediction using the saved embedding:
```
cd src
python main.py
```
Or train from a scratch:
```
cd src
python main.py --train_from_scratch
```

## Running time
For generating random walks (10 walks per node, length 25): **1'15''**

For learning embeddings (100 epoch): **16'12''**

Testing device: laptop, intel i7, nvidia 2060


## File
- README.md
- requirements.txt
- submission.csv (The link prediction result)
- data
  - course3_edge.csv
  - course3_test.csv
  - embedding_128 (The saved embedding file)
- src
  - main.py
  - utils.py
  - encoder.py
  
## Experiment setting
Generate 10 bias random walks of length 25 for each node.

100 epochs. For each epoch, randomly extract 1/10 from the generated random walks for gradient descent. lr = 0.01, and decays by half every 25 epoch. weight_decay = 1e-4 (seems to have little effect).

The final reported value is calculated by 
$$p_{uv}=0.5+0.5\times\dfrac{<e_u, e_v>}{||e_u||||e_v||}$$
**Note** In this deepwalk model, we never model the link probability of $u$ and $v$. At most, we only modeled the co-occurrence probability $p(v|u)$. Therefore, none of $<e_u, e_v>$, $sigmoid(<e_u,e_v>)$ or $softmax(<e_u,e_v>)$ can are reasonable probabilities. They are only scores. The above equation is just a scoreing method used in this problem, considering the accuracy, discrimination and value range. We should train another binary discrimination model to model the link probability, if we want to get statistically reasonable probabilites.

