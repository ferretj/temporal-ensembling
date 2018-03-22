# Temporal Ensembling (PyTorch)

This is the code to reproduce the experiments of my [blog post](https://ferretj.github.io/ml/2018/01/22/temporal-ensembling.html), which explains and gives implementation details on [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/pdf/1610.02242.pdf) from ICLR 2017.

Accuracy on weakly-supervised MNIST (100 labels, 5 seed restarts) : 97.8% (+/- 0.6%).

Best seed accuracy : 98.38%.

## Usage

First, install the requirements in a virtual environment :

```sh
pip install -r requirements.txt
```

Install PyTorch and torchvision as shown [here](http://pytorch.org/) according to your specs.

You can launch a MNIST evaluation from the command line using :

```sh
python mnist_eval.py
```

You can tweak hyperparameters in the config.py file.

## Misc

This code is not a 100% faithful reproduction of the original paper and should not be used as such.

The Theano-based code released by the paper authors can be found [here](https://github.com/smlaine2/tempens).