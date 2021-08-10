The original repo is [here](https://github.com/malllabiisc/HyperGCN)



#Difference with paper

```
paper uses hyper parameter as
hidden layer size : 32
dropout rate : 0.5
learning rate : 0.01
weight decay : o.ooo5
number of training epochs : 200
lamda for explicit Laplacian regularisation : 0.001
Following a prior work Kipf and Welling [2017]

However in condition, HyperGCn does not work well. -> use epoch 2000

Need to find proper hyperparameter -> grid experiments

```
