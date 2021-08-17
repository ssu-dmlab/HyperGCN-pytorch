
import torch, os, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F

from torch.autograd import Variable
from tqdm import tqdm #상태진행률


def train(HyperGCN, dataset, T, args):
    """
    train for a certain number of epochs

    arguments:
	HyperGCN: a dictionary containing model details (gcn, optimiser)
	dataset: the entire dataset
	T: training indices
	args: arguments

	returns:
	the trained model
    """

    hypergcn, optimiser = HyperGCN['model'], HyperGCN['optimiser']
    hypergcn.train()

    X, Y = dataset['features'], dataset['labels']

    for epoch in tqdm(range(args.epochs)):
        optimiser.zero_grad()
        Z = hypergcn(X)
        loss = F.nll_loss(Z[T], Y[T])

        loss.backward()
        optimiser.step()

        # print(accuracy(Z[T], Y[T]))
        # print(test(HyperGCN, dataset, T, args))

    HyperGCN['model'] = hypergcn
    return HyperGCN