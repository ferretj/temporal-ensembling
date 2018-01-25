import numpy as np

# global vars
n_exp        = 5
k            = 100
# model vars
drop         = 0.5
std          = 0.15
fm1          = 32
fm2          = 64
w_norm       = True
# optim vars
lr           = 0.002
beta2        = 0.99
num_epochs   = 300
batch_size   = 100
# temporal ensembling vars
alpha        = 0.6
data_norm    = 'channelwise'
divide_by_bs = False
# RNG
rng          = np.random.RandomState(42)
seeds        = [rng.randint(200) for _ in xrange(n_exp)]