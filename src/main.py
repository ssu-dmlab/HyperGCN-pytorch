from config import config
args = config.parse()

import os, torch, numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import data
'''dataset, trainIndex, test = data.load(args) #load data,dataset parse해서 가져옴 dict형식
print("length of train is", len(trainIndex))
print("length of test is", len(test))
print("dropout is {}".format(args.dropout))'''


import torch.optim as optim, utils
from torch.autograd import Variable
import scipy.sparse as sp

from models import train as T
from models import eval as E

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
    V, E = dataset['n'], dataset['hypergraph'] #V(논문) feature값  E [논문]값
    X, Y = dataset['features'], dataset['labels']  # X=feature matrix Y=label vector

    print(X.shape)
    print(Y.shape)

    # hypergcn and optimiser
    args.d, args.c = X.shape[1], Y.shape[1] #args.d = feature size args.c = label size

    if args.model == "HyperGCN":
        from models.HyperGCN import model
        hypergcn = model.HyperGCN(V, E, X, args)
    elif args.model == "researchGCN":
        from models.researchGCN import model
        hypergcn = model.HyperGCN(V, E, X, args)

    '''
    model = __import__('models.' + args.model + '.model')
    hypergcn = model.HyperGCN(V, E, X, args)
    '''

    optimiser = optim.Adam(list(hypergcn.parameters()), lr=args.rate, weight_decay=args.decay)  # optimiser adam used

    # node features in sparse representation
    X = sp.csr_matrix(utils.normalise(np.array(X)), dtype=np.float32)  # X normalize
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


dataset, trainIndex, test = data.load(args)  # load data,dataset parse해서 가져옴 dict형식
print("length of train is", len(trainIndex))
print("length of test is", len(test))
print("dropout is {}".format(args.dropout))

#step 0. Initialization, Load datasets
HyperGCN = initialise(dataset, args)

#step 1. Run (train and evaluate) the specified model

HyperGCN = T.train(HyperGCN, dataset, trainIndex, args)  #model.py train function
acc = E.test(HyperGCN, dataset, test, args)         #model.py test function

#step 2. Reprotr and save the final results
print("accuracy:", float(acc), ", error:", float(100*(1-acc))) #model.py accuracy, error
