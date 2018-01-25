from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gsp
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as tf


class GaussianNoise(nn.Module):
    
    def __init__(self, batch_size, input_shape=(1, 28, 28), std=0.05):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape).cuda())
        self.std = std
        
    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        return x + self.noise


def prepare_mnist():
    # normalize data
    m = (0.1307,)
    st = (0.3081,)
    normalize = tf.Normalize(m, st)
        
    # load train data
    train_dataset = datasets.MNIST(
                        root='../data', 
                        train=True, 
                        transform=tf.Compose([tf.ToTensor(), normalize]),  
                        download=True)
    
    # load test data
    test_dataset = datasets.MNIST(
                        root='../data', 
                        train=False, 
                        transform=tf.Compose([tf.ToTensor(), normalize]))
    
    return train_dataset, test_dataset


def ramp_up(epoch, max_epochs, max_val, mult):
    if epoch == 0:
        return 0.
    elif epoch >= max_epochs:
        return max_val
    return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)


def weight_schedule(epoch, max_epochs, max_val, mult, n_labeled, n_samples):
    max_val = max_val * (float(n_labeled) / n_samples)
    return ramp_up(epoch, max_epochs, max_val, mult)


def calc_metrics(model, loader):
    correct = 0
    total = 0
    for i, (samples, labels) in enumerate(loader):
        samples = Variable(samples.cuda(), volatile=True)
        labels = Variable(labels.cuda())
        outputs = model(samples)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data.view_as(predicted)).sum()

    acc = 100 * float(correct) / total
    return acc


def savetime():
    return datetime.now().strftime('%Y_%m_%d_%H%M%S')


def save_losses(losses, sup_losses, unsup_losses, fname, labels=None):
    plt.style.use('ggplot')
    
    # color palette from Randy Olson
    colors = [
        (31, 119, 180),
        (174, 199, 232),
        (255, 127, 14), 
        (255, 187, 120),    
        (44, 160, 44),
        (152, 223, 138),
        (214, 39, 40),
        (255, 152, 150),    
        (148, 103, 189),
        (197, 176, 213), 
        (140, 86, 75),
        (196, 156, 148),    
        (227, 119, 194),
        (247, 182, 210),
        (127, 127, 127),
        (199, 199, 199),    
        (188, 189, 34),
        (219, 219, 141),
        (23, 190, 207),
        (158, 218, 229)]

    colors = [(float(c[0]) / 255, float(c[1]) / 255, float(c[2]) / 255) for c in colors]

    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    for i in xrange(3):
        axs[i].tick_params(axis="both", which="both", bottom="off", top="off",    
                           labelbottom="on", left="off", right="off", labelleft="on")
    for i in xrange(len(losses)):
        axs[0].plot(losses[i], color=colors[i])
        axs[1].plot(sup_losses[i], color=colors[i])
        axs[2].plot(unsup_losses[i], color=colors[i])
    axs[0].set_title('Overall loss', fontsize=14)
    axs[1].set_title('Supervised loss', fontsize=14)
    axs[2].set_title('Unsupervised loss', fontsize=14)
    if labels is not None:
        axs[0].legend(labels)
        axs[1].legend(labels)
        axs[2].legend(labels)
    plt.savefig(fname)


def save_exp(time, losses, sup_losses, unsup_losses,
             accs, accs_best, idxs, **kwargs):
    
    def save_txt(fname, accs, **kwargs):
        with open(fname, 'w') as fp:
            fp.write('GLOB VARS\n')
            fp.write('n_exp        = {}\n'.format(kwargs['n_exp']))
            fp.write('k            = {}\n'.format(kwargs['k']))
            fp.write('MODEL VARS\n')
            fp.write('drop         = {}\n'.format(kwargs['drop']))
            fp.write('std          = {}\n'.format(kwargs['std']))
            fp.write('fm1          = {}\n'.format(kwargs['fm1']))
            fp.write('fm2          = {}\n'.format(kwargs['fm2']))
            fp.write('w_norm       = {}\n'.format(kwargs['w_norm']))
            fp.write('OPTIM VARS\n')
            fp.write('lr           = {}\n'.format(kwargs['lr']))
            fp.write('beta2        = {}\n'.format(kwargs['beta2']))
            fp.write('num_epochs   = {}\n'.format(kwargs['num_epochs']))
            fp.write('batch_size   = {}\n'.format(kwargs['batch_size']))
            fp.write('TEMP ENSEMBLING VARS\n')
            fp.write('alpha        = {}\n'.format(kwargs['alpha']))
            fp.write('data_norm    = {}\n'.format(kwargs['data_norm']))
            fp.write('divide_by_bs = {}\n'.format(kwargs['divide_by_bs']))
            fp.write('\nRESULTS\n')
            fp.write('best accuracy : {}\n'.format(np.max(accs)))
            fp.write('accuracy : {} (+/- {})\n'.format(np.mean(accs), np.std(accs)))
            fp.write('accs : {}\n'.format(accs))
        
    labels = ['seed_' + str(sd) for sd in kwargs['seeds']]
    if not os.path.isdir('exps'):
        os.mkdir('exps')
    time_dir = os.path.join('exps', time)
    if not os.path.isdir(time_dir):
        os.mkdir(time_dir)
    fname_bst = os.path.join('exps', time, 'training_best.png')
    fname_fig = os.path.join('exps', time, 'training_all.png')
    fname_smr = os.path.join('exps', time, 'summary.txt')
    fname_sd  = os.path.join('exps', time, 'seed_samples')
    best = np.argmax(accs_best)
    save_losses([losses[best]], [sup_losses[best]], [unsup_losses[best]], fname_bst)
    save_losses(losses, sup_losses, unsup_losses, fname_fig, labels=labels)
    for seed, indices in zip(kwargs['seeds'], idxs):
        save_seed_samples(fname_sd + '_seed' + str(seed) + '.png', indices)
    save_txt(fname_smr, accs_best, **kwargs)


def save_seed_samples(fname, indices):
    train_dataset, test_dataset = prepare_mnist()
    imgs = train_dataset.train_data[indices.numpy().astype(int)]
    
    plt.style.use('classic')
    fig = plt.figure(figsize=(15, 60))
    gs = gsp.GridSpec(20, 5, width_ratios=[1, 1, 1, 1, 1],
                      wspace=0.0, hspace=0.0)
    for ll in xrange(100):
        i = ll // 5
        j = ll % 5
        img = imgs[ll].numpy()
        ax = plt.subplot(gs[i, j])
        ax.tick_params(axis="both", which="both", bottom="off", top="off",
                       labelbottom="off", left="off", right="off", labelleft="off")
        ax.imshow(img)
    
    plt.savefig(fname)
