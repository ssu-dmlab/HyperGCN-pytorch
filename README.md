The original repo is [here](https://github.com/malllabiisc/HyperGCN)



#Difference with paper

```
paper uses hyperparameter as
hidden layer size : 32
dropout rate : 0.5
learning rate : 0.01
weight decay : o.ooo5
number of training epochs : 200
lamda for explicit Laplacian regularisation : 0.001
Following a prior work Kipf and Welling [2017]

However in condition, HyperGCN does not work well. -> use epoch 2000

Need to find proper hyperparameter -> grid experiments (about learning rate, epoch in each split) 

Proper hyperparameter

hidden layer size : 32
dropout rate : 0.5
learning rate : 0.05
weight decay : o.ooo5
number of training epochs : 500
lamda for explicit Laplacian regularisation : 0.001

split  dp  epoch  acc
1 	  0.05	500	  0.8555431980183000
2 	  0.05	500	  0.8527627521358880
3 	  0.05	500	  0.8494262170769930
4   	  0.05	500	  0.8536979930236090
5	  0.05	500	  0.8523330468631520
6   	  0.05	500	  0.8436378342854250
7	  0.05	500	  0.8463677266063390
8	  0.05	500	  0.8418937364137300
9.        0.05	500	  0.8411859865527530
10  	  0.05	500	  0.8469996461250700

else (lr, epoch) = {(0.05, 2000), (0.1,1000)} has similiar result as above
```
