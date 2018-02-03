import os, sys
sys.path.append(os.getcwd())

import time
import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.cifar10
import tflib.plot
# import tflib.inception_score

import numpy as np


import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = '/mnt/dataset2'
if not os.path.exists(DATA_DIR):
    DATA_DIR = '/workspace/data/cifar-10-batches-py'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

if not os.path.exists('/results/cifar10'):
    os.makedirs('/results/cifar10')

# MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp

mode = str(sys.argv[1])
print('Mode: ' + mode)

DIM = 128 # This overfits substantially; you're probably better off with 64
# LAMBDA = 10 # Gradient penalty lambda hyperparameter
LAMBDA = float(sys.argv[2])
# CRITIC_ITERS = 5 # How many critic iterations per generator iteration
CRITIC_ITERS = int(sys.argv[3])
BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

B1 = float(sys.argv[4])
print('B1: ' + str(B1))

LR = float(sys.argv[5])
print('LR: ' + str(LR))

Reg_on = str(sys.argv[6])

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.BatchNorm2d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output

netG = Generator()
netD = Discriminator()
# print(netG)
# print(netD)

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(B1, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(B1, 0.9))

def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous().view(BATCH_SIZE, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# For generating samples
def generate_image(frame, netG):
    fixed_noise_128 = torch.randn(128, 128)
    if use_cuda:
        fixed_noise_128 = fixed_noise_128.cuda(gpu)
    noisev = autograd.Variable(fixed_noise_128, volatile=True)
    samples = netG(noisev)
    samples = samples.view(-1, 3, 32, 32)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(samples, '/results/cifar10/samples_{}.jpg'.format(frame))

# For calculating inception score
# def get_inception_score(G, ):
#     all_samples = []
#     for i in range(10):
#         samples_100 = torch.randn(100, 128)
#         if use_cuda:
#             samples_100 = samples_100.cuda(gpu)
#         samples_100 = autograd.Variable(samples_100, volatile=True)
#         all_samples.append(G(samples_100).cpu().data.numpy())

#     all_samples = np.concatenate(all_samples, axis=0)
#     all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
#     all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
#     return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterator
train_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)
def inf_train_gen():
    while True:
        for images in train_gen():
            # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            yield images
gen = inf_train_gen()
preprocess = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

if mode != 'wgp':
    criterion = nn.BCEWithLogitsLoss()
    label = torch.FloatTensor(BATCH_SIZE)
    if use_cuda:
        criterion.cuda()
        label = label.cuda()

svds = {}
for name, param in netG.named_parameters():
    if 'bias' not in name:
        svds['G.' + name] = []
for name, param in netD.named_parameters():
    if 'bias' not in name:
        svds['D.' + name] = []

for iteration in range(ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for i in range(CRITIC_ITERS):
        import ipdb; ipdb.set_trace()
        _data = gen.__next__()
        netD.zero_grad()

        # train with real
        _data = _data.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
        real_data = torch.stack([preprocess(item) for item in _data])

        if use_cuda:
            real_data = real_data.cuda(gpu)
        real_data_v = autograd.Variable(real_data)

        if mode == 'reg' or mode == 'gp' or ('dwd' in mode):
            label.resize_(BATCH_SIZE, 1).fill_(1)
            labelv = autograd.Variable(label)
            output = netD(real_data_v)
            D_cost_real = criterion(output, labelv)
            D_cost_real.backward(retain_graph=True)
        if mode == 'wgp':
            D_cost_real = netD(real_data_v)
            D_cost_real = D_cost_real.mean()
            D_cost_real.backward(mone)

        # D_real = netD(real_data_v)
        # D_real = D_real.mean()
        # D_real.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        fake = autograd.Variable(netG(noisev).data)
        inputv = fake

        if mode == 'reg' or mode == 'gp' or ('dwd' in mode):
            label.resize_(BATCH_SIZE, 1).fill_(0)
            labelv = autograd.Variable(label)
            output = netD(inputv)
            D_cost_fake = criterion(output, labelv)
            D_cost_fake.backward(retain_graph=True)
        if mode == 'wgp':
            D_cost_fake = netD(inputv)
            D_cost_fake = D_cost_fake.mean()
            D_cost_fake.backward(one)

        # D_fake = netD(inputv)
        # D_fake = D_fake.mean()
        # D_fake.backward(one)

        if mode == 'wgp' or mode == 'gp':
            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
            gradient_penalty.backward()

        if ('dwd' in mode):
            # grads = autograd.grad(D_cost_real + D_cost_fake, netD.parameters())
            list_weights = []
            for name, param in netD.named_parameters():
                if 'bias' not in name:
                    list_weights += [param]
                # if 'conv1d' in name:
                #     list_weights += [param]

            grads = autograd.grad(
                outputs=D_cost_real + D_cost_fake,
                inputs=list_weights,
                grad_outputs=torch.ones((D_cost_real + D_cost_fake).size()).cuda() if use_cuda else torch.ones(
                    (D_cost_real + D_cost_fake).size()),
                create_graph=True, retain_graph=True, only_inputs=True)

            denoms = [torch.sum(g ** 2) for g in grads]
            noms = [torch.sum(torch.mm(
                g.view(g.size()[0], -1),
                g.view(g.size()[0], -1).transpose(0, 1)) ** 2) for g in grads]

            if '1' in mode:
                pen = LAMBDA * sum([torch.sum(g ** 2) for g in grads])
            if '2' in mode:
                pen = LAMBDA * sum([n / (d ** 2 + 1e-8) for n, d in zip(noms, denoms)])
            pen.backward()

        D_cost = D_cost_real + D_cost_fake
        optimizerD.step()
    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake = netG(noisev)

    if mode == 'reg' or mode == 'gp' or ('dwd' in mode):

        label.resize_(BATCH_SIZE, 1).fill_(1)
        labelv = autograd.Variable(label)
        output = netD(fake)
        G_cost = criterion(output, labelv)
        G_cost.backward(retain_graph=True)

    if mode == 'wgp':
        G_cost = netD(fake)
        G_cost = G_cost.mean()
        G_cost.backward(mone)
        G_cost = -G_cost

    if ('dwd' in mode) and ('both' in Reg_on):
        list_weights = []
        for name, param in netG.named_parameters():
            if 'bias' not in name:
                list_weights += [param]

        grads = autograd.grad(
            outputs=G_cost,
            inputs=list_weights,
            grad_outputs=torch.ones((G_cost).size()).cuda() if use_cuda else torch.ones(
                (G_cost).size()),
            create_graph=True, retain_graph=True, only_inputs=True)

        pen = LAMBDA * sum([torch.sum(g ** 2) for g in grads]) / 2.0
        pen.backward()

    # Calculate dev loss and generate samples every 100 iters
    if iteration % 100 == 99:

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

        if mode == 'wgp' or mode == 'gp' or mode == 'reg':
            print('iter: ' + str(iteration) + ', ' +
                  'G_cost: ' + str(G_cost.cpu().data.numpy()) + ', ' +
                  'D_cost: ' + str(D_cost.cpu().data.numpy()) + ', ')
        if ('dwd' in mode):
            print('iter: ' + str(iteration) + ', ' +
                  'G_cost: ' + str(G_cost.cpu().data.numpy()) + ', ' +
                  'D_cost: ' + str(D_cost.cpu().data.numpy()) + ', ' +
                  'pen: ' + str(pen.cpu().data.numpy()))

        generate_image(iteration, netG)

    if iteration % 1000 == 999:
        print('SVDS saved!')
        np.save('/results/cifar10/svds', svds)
        # svdplot_cifar('/results/cifar10/')

    # Save logs every 100 iters
    if (iteration < 5) or (iteration % 100 == 99):
        lib.plot.flush()
    lib.plot.tick()
