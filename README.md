# Temporal Ensembling (PyTorch)

This is the code to reproduce the experiments of my [blog post](https://ferretj.github.io/jekyll/update/2018/01/22/temporal-ensembling.html), which explains and gives implementation details on [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/pdf/1610.02242.pdf) from ICLR 2017.

Results on weakly-supervised MNIST (100 labels) : 97.8 (+/- 0.6)

## Usage

First, install the requirements in a virtual environment.

Install PyTorch and torchvision as explained on the [PyTorch website](http://pytorch.org/)

Then you can execute the command :

```py
python mnist_eval.py
```

## Misc

This code is not a 100% faithful reproduction of the original paper and should not be used as such.

The code released by the paper authors can be found [here](https://github.com/smlaine2/tempens). It uses Theano.