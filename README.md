# HyperGCN
_________
This is Pytorch implementation of **[HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs](https://github.com/malllabiisc/HyperGCN)** (NeurIPS 2019).

## Requirements
______________
> tqdm == 4.62.1  
> torch == 1.9.0  
> scipy == 1.7.1  
> numpy == 1.21.2  
> ConfigArgParse == 1.5.2  

install the requirements using `pip install -r requirements.txt`

## Dependencies
* Datasets : For data (and/or splits) not used in the paper
> coauthorship : dblp, cora  
> cocitation : citeseer, cora, pubmed
* Hyperparameters : used in the paper - Following a prior work **[Kipf and Welling](https://github.com/tkipf/gcn)** (2017)
> hidden layer size : 32  
> dropout rate : 0.5  
> learning rate : 0.01  
> weight decay : o.ooo5  
> training epochs : 200  
> lamda for explicit Laplacian regularisation : 0.001  

## Training
* Purpose : `Hypernode classification`
* To start training run :  
    `python main.py --mediators True --split 1 --data coauthorship -- dataset dblp`

In the project, config.py has hyperparameters as follows :

 Option | Description | Default
 ------- | ---------- | --------
 data | oauthorship/cocitation | coauthorship
 dataset | cora/dblp for coauthorship, cora/citeseer/pubmed for cocitation | dblp
 mediators | use of Laplacian | False
 fast | use of FastHyperGCN | False
 split | train-test split dataset | 1
 gpu | gpu number to use | 3
 cuda | True or False | True
 seed | and integer for random | 5
 depth | number of hidden layers | 2
 dropout | dropout probability for hidden layers | 0.5
 epochs | number of training epochs | 200
 rate | learning rate | 0.01
 decay | weight decay | 0.0005
 model | model for hypergraph | HyperGCN

## Difference with paper

However in this default condition, HyperGCN does not train well.  
- Check **[issue](https://github.com/malllabiisc/HyperGCN/issues/1)** about following result
###Experiments of 1. coauthorship/dblp, 2. coauthorship/cora

![coauthorship/dblp.png](coauthorship-dblp.png) ![coauthorship/cora.png](coauthorship-cora.png)

So we need to find proper hyperparameter -> `Grid experiments` (about learning rate, epoch in each split without validation part)
##Experiment's parameters of coauthorship/dblp
> Split = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
> Learning rate = [0.001, 0.005, 0.01, 0.05, 0.1]  
> Epoch = [100, 200, 500, 1000, 2000]  

###High accuracy result

 Split | Learning rate | Epochs | Accuracy 
 ----- | ------------- | ------ | --------
 1 | 0.1 | 2000 | 0.8548101713765730
 2 | 0.1 | 2000 | 0.8494262170769930
 3 | 0.1 | 2000 | 0.8577928315049800
 4 | 0.1 | 2000 | 0.852889136039634
 5 | 0.1 | 2000 | 0.8527880289166370
 6 | 0.1 | 2000 | 0.8491987260502500
 7 | 0.1 | 2000 | 0.847429351397806
 8 | 0.1 | 2000 | 0.8456346999646130
 9 | 0.1 | 2000 | 0.840983772306759
 10 | 0.1 | 2000 | 0.8445477983923970

- `mean of accuracy` is 0.849550073302664 ~ 0.85  
- {Learning rate , Epoch} = [(0.05,500), (0.05,2000), (0.1,1000)] has similiar result as above
