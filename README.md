# Temporal Ensembling (PyTorch)

This is the code to reproduce the experiments of my [blog post](https://ferretj.github.io/ml/2018/01/22/temporal-ensembling.html), which explains and gives implementation details on [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/pdf/1610.02242.pdf) from ICLR 2017.

Accuracy on weakly-supervised MNIST (100 labels, 5 seed restarts) : 97.8% (+/- 0.6%).

Best seed accuracy : 98.38%.

## Usage

#### Standard requirements

First, install the requirements in a virtual environment :

```sh
pip install -r requirements.txt
```

#### Regarding PyTorch and torchvision

I used PyTorch version 0.3.0.post4 and torchvision version 0.2.0, so these are the recommended versions.

If you want to run it using PyTorch 0.4+, see [this issue](https://github.com/ferretj/temporal-ensembling/issues/1). 

Install PyTorch and torchvision as shown [here](http://pytorch.org/) according to your specs.

#### Training a model

You can launch a MNIST evaluation from the command line using :

```sh
python mnist_eval.py
```

You can tweak hyperparameters in the config.py file.

## Misc

This code is not a 100% faithful reproduction of the original paper and should not be used as such.

The Theano-based code released by the paper authors can be found [here](https://github.com/smlaine2/tempens).