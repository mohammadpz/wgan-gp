import os, sys

sys.path.append(os.getcwd())

import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import tflib as lib
import tflib.plot

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

mode = str(sys.argv[1])
print('Mode: ' + mode)

MODE = 'wgan-gp'  # wgan or wgan-gp
DATASET = '8gaussians'  # 8gaussians, 25gaussians, swissroll
DIM = 512  # Model dimensionality
FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
# Gaussian noise, as in the plots in the paper
# LAMBDA = 0.0001  # Smaller lambda seems to help for toy tasks specifically
LAMBDA = float(sys.argv[3])
# CRITIC_ITERS = 5  # How many critic iterations per generator iteration
CRITIC_ITERS = int(sys.argv[2])
BATCH_SIZE = 256  # Batch size
ITERS = 100000  # how many generator iterations to train for
use_cuda = True

# ==================Definition Start======================

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 2),
        )
        self.main = main

    def forward(self, noise, real_data):
        if FIXED_GENERATOR:
            return noise + real_data
        else:
            output = self.main(noise)
            return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Linear(2, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 1)
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

frame_index = [0]
def generate_image(true_dist):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    points_v = autograd.Variable(torch.Tensor(points), volatile=True)
    if use_cuda:
        points_v = points_v.cuda()
    disc_map = netD(points_v).cpu().data.numpy()

    noise = torch.randn(BATCH_SIZE, 2)
    if use_cuda:
        noise = noise.cuda()
    noisev = autograd.Variable(noise, volatile=True)
    true_dist_v = autograd.Variable(torch.Tensor(true_dist).cuda() if use_cuda else torch.Tensor(true_dist))
    samples = netG(noisev, true_dist_v).cpu().data.numpy()

    plt.clf()

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
    if not FIXED_GENERATOR:
        plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    plt.savefig('/results/' + DATASET + '_' + mode + '/' + 'frame' + str(frame_index[0]) + '.jpg')

    frame_index[0] += 1


# Dataset iterator
def inf_train_gen():
    if DATASET == '25gaussians':

        dataset = []
        for i in range(100000 / 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828  # stdev
        while True:
            for i in range(len(dataset) / BATCH_SIZE):
                yield dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

    elif DATASET == 'swissroll':

        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=BATCH_SIZE,
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5  # stdev plus a little
            yield data

    elif DATASET == '8gaussians':

        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                point = np.random.randn(2) * .02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414  # stdev
            yield dataset


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def calc_dwd(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# ==================Definition End======================

netG = Generator()
netD = Discriminator()
netD.apply(weights_init)
netG.apply(weights_init)
print(netG)
print(netD)

if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

data = inf_train_gen()

if mode != 'wgp':
    criterion = nn.BCEWithLogitsLoss()
    label = torch.FloatTensor(BATCH_SIZE)
    if use_cuda:
        criterion.cuda()
        label = label.cuda()

if not os.path.exists('/results/' + DATASET + '_' + mode):
    os.makedirs('/results/' + DATASET + '_' + mode)

svds = {}
for name, param in netG.named_parameters():
    if 'bias' not in name:
        svds['G.' + name] = []
for name, param in netD.named_parameters():
    if 'bias' not in name:
        svds['D.' + name] = []

print('/results/' + DATASET + '_' + mode)
for iteration in range(ITERS):
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in range(CRITIC_ITERS):
        _data = data.__next__()
        real_data = torch.Tensor(_data)
        if use_cuda:
            real_data = real_data.cuda()
        real_data_v = autograd.Variable(real_data)

        netD.zero_grad()

        # train with real
        # if mode == 'dwd':
        #     label.resize_(BATCH_SIZE).fill_(1)
        #     labelv = autograd.Variable(label)
        #     output = netD(real_data_v)
        #     D_cost_real = criterion(output, labelv)
        #     grads = autograd.grad(D_cost_real, netD.parameters())
        #     pen = sum([torch.sum(g ** 2) for g in grads])
        #     D_cost_real = D_cost_real + pen
        #     D_cost_real.backward()
        if mode == 'reg' or mode == 'gp' or mode == 'dwd':
            label.resize_(BATCH_SIZE).fill_(1)
            labelv = autograd.Variable(label)
            output = netD(real_data_v)
            D_cost_real = criterion(output, labelv)
            D_cost_real.backward(retain_graph=True)
        if mode == 'wgp':
            D_cost_real = netD(real_data_v)
            D_cost_real = D_cost_real.mean()
            D_cost_real.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, 2)
        if use_cuda:
            noise = noise.cuda()
        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        fake = autograd.Variable(netG(noisev, real_data_v).data)
        inputv = fake
        # if mode == 'dwd':
        #     label.resize_(BATCH_SIZE).fill_(0)
        #     labelv = autograd.Variable(label)
        #     output = netD(inputv)
        #     D_cost_fake = criterion(output, labelv)
        #     grads = autograd.grad(D_cost_fake, netD.parameters())
        #     pen = sum([torch.sum(g ** 2) for g in grads])
        #     D_cost_fake = D_cost_fake + pen
        #     D_cost_fake.backward()

        if mode == 'reg' or mode == 'gp' or mode == 'dwd':
            label.resize_(BATCH_SIZE).fill_(0)
            labelv = autograd.Variable(label)
            output = netD(inputv)
            D_cost_fake = criterion(output, labelv)
            D_cost_fake.backward(retain_graph=True)
        if mode == 'wgp':
            D_cost_fake = netD(inputv)
            D_cost_fake = D_cost_fake.mean()
            D_cost_fake.backward(one)

        if mode == 'wgp' or mode == 'gp':
            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
            gradient_penalty.backward()

        if mode == 'dwd':
            # grads = autograd.grad(D_cost_real + D_cost_fake, netD.parameters())
            grads = autograd.grad(
                outputs=D_cost_real + D_cost_fake,
                inputs=netD.parameters(),
                grad_outputs=torch.ones((D_cost_real + D_cost_fake).size()).cuda() if use_cuda else torch.ones(
                    (D_cost_real + D_cost_fake).size()),
                create_graph=True, retain_graph=True, only_inputs=True)
            pen = LAMBDA * sum([torch.sum(g ** 2) for g in grads])
            pen.backward()

        optimizerD.step()

        D_cost = D_cost_fake + D_cost_real

    if not FIXED_GENERATOR:
        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()

        _data = data.__next__()
        real_data = torch.Tensor(_data)
        if use_cuda:
            real_data = real_data.cuda()
        real_data_v = autograd.Variable(real_data)

        noise = torch.randn(BATCH_SIZE, 2)
        if use_cuda:
            noise = noise.cuda()
        noisev = autograd.Variable(noise)
        fake = netG(noisev, real_data_v)

        if mode == 'reg' or mode == 'gp' or mode == 'dwd':
            label.resize_(BATCH_SIZE).fill_(1)
            labelv = autograd.Variable(label)
            output = netD(fake)
            G_cost = criterion(output, labelv)
            G_cost.backward()

        if mode == 'wgp':
            G_cost = netD(fake)
            G_cost = G_cost.mean()
            G_cost.backward(mone)
            G_cost = -G_cost
        optimizerG.step()

    for name, param in netG.named_parameters():
        if 'bias' not in name:
            p = param.cpu().data.numpy()
            svds['G.' + name] += [np.linalg.svd(
                p.reshape((p.shape[0], -1)),
                full_matrices=False, compute_uv=False)]
    for name, param in netD.named_parameters():
        if 'bias' not in name:
            p = param.cpu().data.numpy()
            svds['D.' + name] += [np.linalg.svd(
                p.reshape((p.shape[0], -1)),
                full_matrices=False, compute_uv=False)]

    # Write logs and save samples
    lib.plot.plot('/results/' + DATASET + '_' + mode + '/' + 'D_cost', D_cost.cpu().data.numpy())
    if not FIXED_GENERATOR:
        lib.plot.plot('/results/' + DATASET + '_' + mode + '/' + 'G_cost', G_cost.cpu().data.numpy())
    if iteration % 100 == 0:
        lib.plot.flush()
        generate_image(_data)

        print('SVDS saved!')
        np.save('/results/' + DATASET + '_' + mode + '/svds', svds)

    lib.plot.tick()
