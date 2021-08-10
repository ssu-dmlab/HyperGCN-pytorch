from config import config
args = config.parse()

import os, torch, numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import data
dataset, train, test = data.load(args) #load data,dataset parse해서 가져옴 dict형식
print("length of train is", len(train))
print("length of test is", len(test))
print("how can make val part?")
print("dropout is {}".format(args.dropout))


def initialise(dataset, args):
    """
    initialises GCN, optimiser, normalises graph, and features, and sets GPU number

    arguments:
    dataset: the entire dataset (with graph, features, labels as keys)
    args: arguments

    returns:
    a dictionary with model details (hypergcn, optimiser)
    """

    HyperGCN = {}
    V, E = dataset['n'], dataset['hypergraph']
    X, Y = dataset['features'], dataset['labels']  # X=feature matrix Y=label vector

    # hypergcn and optimiser
    args.d, args.c = X.shape[1], Y.shape[1]
    hypergcn = networks.HyperGCN(V, E, X, args)
    optimiser = optim.Adam(list(hypergcn.parameters()), lr=args.rate, weight_decay=args.decay)  # optimiser adam used

    # node features in sparse representation
    X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)  # X normalize
    X = torch.FloatTensor(np.array(X.todense()))  # make X np tensor

    # labels
    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1])  # 열벡터 변환

    # cuda
    args.Cuda = args.cuda and torch.cuda.is_available()
    if args.Cuda:
        hypergcn.cuda()
        X, Y = X.cuda(), Y.cuda()

    # update dataset with torch autograd variable
    dataset['features'] = Variable(X)  # what is variable function?
    dataset['labels'] = Variable(Y)

    # update model and optimiser
    HyperGCN['model'] = hypergcn
    HyperGCN['optimiser'] = optimiser
    return HyperGCN


def normalise(M):  # normalize matrix
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)  # D inverse i.e. D^{-1}

    return DI.dot(M)

#step 0. Initialization, Load datasets
HyperGCN = initialise(dataset, args)

#step 1. Run (train and evaluate) the specified model
from model import model
HyperGCN = model.train(HyperGCN, dataset, train, args)  #model.py train function
acc = model.test(HyperGCN, dataset, test, args)         #model.py test function

#step 2. Reprotr and save the final results
print("accuracy:", float(acc), ", error:", float(100*(1-acc))) #model.py accuracy, error

