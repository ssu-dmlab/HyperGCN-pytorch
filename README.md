# HyperGCN

This repository aims to reproduce **HyperGCN** proposed in the paper entitled "HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs (NeurIPS 2019)". 
We refer to [the original repository](https://github.com/malllabiisc/HyperGCN) of **HyperGCN** to implement this repository. 

## Dependencies

We use the following Python packages to implement this. Install them in your environment using `pip install -r requirements.txt`. 

* tqdm == 4.62.1  
* torch == 1.9.0  
* scipy == 1.7.1  
* numpy == 1.21.2  
* ConfigArgParse == 1.5.2  

## Experimental setting

### Datasets
We use the following datasets for the experiments. 
* coauthorship : dblp, cora  
* cocitation : citeseer, cora, pubmed

### Hyperparameters
We use the following hyperparameters as described in **[Kipf and Welling](https://github.com/tkipf/gcn)**. 
* hidden layer size : 32  
* dropout rate : 0.5  
* learning rate : 0.01  
* weight decay : 0.0005  
* training epochs : 200  
* lamda for explicit Laplacian regularisation : 0.001  

## Training

To train the model, type the following command: 
```bash
python main.py --mediators True --split 1 --data coauthorship -- dataset dblp
```

It will train the model for the node classification task in the given hypergraph. 

In this project, you can control the value of each hyperparameter in `config.py`. 
The details of the hyperparameters are in the following table. 

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

## Difference between this repository and the original repository



However in this default condition, HyperGCN does not train well.  
- Check **[issue](https://github.com/malllabiisc/HyperGCN/issues/1)** about following result  

- Experiments of coauthorship/dblp(left), coauthorship/cora(right) in dafault condition

![coauthorship/dblp.png](coauthorship-dblp.png) ![coauthorship/cora.png](coauthorship-cora.png)

So we need to find proper hyperparameter -> `Grid experiments` (about learning rate, epoch in each split without validation part)

## Experiment's parameters of coauthorship/dblp

> Split = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
> Learning rate = [0.001, 0.005, 0.01, 0.05, 0.1]  
> Epoch = [100, 200, 500, 1000, 2000]  

**The highest accuracy result**

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
