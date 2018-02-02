import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import language_helpers
import tflib as lib
import tflib.plot
from svd_plot import svdplot

from sklearn.preprocessing import OneHotEncoder

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

mode = str(sys.argv[1])
print('Mode: ' + mode)

# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!
DATA_DIR = '/mnt/dataset1'
if not os.path.exists(DATA_DIR):
    DATA_DIR = '/workspace/data/1-billion-word-language-modeling-benchmark-r13output'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_language.py!')

BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many iterations to train for
SEQ_LEN = 32 # Sequence length in characters
DIM = 512 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
# CRITIC_ITERS = 10 # How many critic iterations per generator iteration. We
                  # use 10 for the results in the paper, but 5 should work fine
                  # as well.
CRITIC_ITERS = int(sys.argv[3])
# LAMBDA = 10 # Gradient penalty lambda hyperparameter.
LAMBDA = float(sys.argv[2])
# print("LAMBDA: " + str(LAMBDA))
MAX_N_EXAMPLES = 10000000#10000000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).
B1 = float(sys.argv[4])
print('B1: ' + str(B1))

LR = float(sys.argv[5])
print('LR: ' + str(LR))

LMB = float(sys.argv[6])

lib.print_model_settings(locals().copy())

lines, charmap, inv_charmap = language_helpers.load_dataset(
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)

table = np.arange(len(charmap)).reshape(-1, 1)
one_hot = OneHotEncoder()
one_hot.fit(table)

# ==================Definition Start======================

def make_noise(shape, volatile=False):
    tensor = torch.randn(shape).cuda(gpu) if use_cuda else torch.randn(shape)
    return autograd.Variable(tensor, volatile)

class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()

        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(DIM, DIM, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(DIM, DIM, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3 * output)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(128, DIM * SEQ_LEN)
        self.block = nn.Sequential(
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
        )
        self.conv1 = nn.Conv1d(DIM, len(charmap), 1)
        self.softmax = nn.Softmax()

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, DIM, SEQ_LEN) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(BATCH_SIZE*SEQ_LEN, -1)
        output = self.softmax(output)
        return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.block = nn.Sequential(
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
        )
        self.conv1d = nn.Conv1d(len(charmap), DIM, 1)
        self.linear = nn.Linear(SEQ_LEN*DIM, 1)

    def forward(self, input):
        output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, SEQ_LEN*DIM)
        output = self.linear(output)
        return output

# Dataset iterator
def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-BATCH_SIZE+1, BATCH_SIZE):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+BATCH_SIZE]],
                dtype='int32'
            )

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    # TODO: Make ConvBackward diffentiable
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def generate_samples(netG):
    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise, volatile=True)
    samples = netG(noisev)
    samples = samples.view(-1, SEQ_LEN, len(charmap))
    # print samples.size()

    samples = samples.cpu().data.numpy()

    samples = np.argmax(samples, axis=2)
    decoded_samples = []
    for i in range(len(samples)):
        decoded = []
        for j in range(len(samples[i])):
            decoded.append(inv_charmap[samples[i][j]])
        decoded_samples.append(tuple(decoded))
    return decoded_samples

# ==================Definition End======================

netG = Generator()
netD = Discriminator()
# print(netG)
# print(netD)

if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(B1, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(B1, 0.9))
# optimizerD = optim.SGD(netD.parameters(), lr=LR, momentum=B1)
# optimizerG = optim.SGD(netG.parameters(), lr=LR, momentum=B1)

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

data = inf_train_gen()

# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.
# true_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines[10*BATCH_SIZE:], tokenize=False) for i in range(4)]
# validation_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines[:10*BATCH_SIZE], tokenize=False) for i in range(4)]
# for i in range(4):
#     print("validation set JSD for n={}: {}".format(i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
# true_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]

if not os.path.exists('/results/lang_' + mode):
    os.makedirs('/results/lang_' + mode)

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

    for iter_d in range(CRITIC_ITERS):
        _data = data.__next__()
        data_one_hot = one_hot.transform(_data.reshape(-1, 1)).toarray().reshape(BATCH_SIZE, -1, len(charmap))
        real_data = torch.Tensor(data_one_hot)
        if use_cuda:
            real_data = real_data.cuda(gpu)
        real_data_v = autograd.Variable(real_data)

        netD.zero_grad()
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

        # # train with real
        # D_real = netD(real_data_v)
        # D_real = D_real.mean()
        # # print D_real
        # # TODO: Waiting for the bug fix from pytorch
        # D_real.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        fake = autograd.Variable(netG(noisev).data)
        inputv = fake
        # D_fake = netD(inputv)
        # D_fake = D_fake.mean()
        # # TODO: Waiting for the bug fix from pytorch
        # D_fake.backward(one)

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

        if mode == 'wgp' or mode == 'gp':
            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
            gradient_penalty.backward()

        if ('dwd' in mode):
            # grads = autograd.grad(D_cost_real + D_cost_fake, netD.parameters())
            list_weights = []
            for name, param in netD.named_parameters():
                if ('bias' not in name) and ('conv1d' in name):
                    list_weights += [param]
            assert len(list_weights) == 1

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
                pen = LAMBDA * sum([torch.sum(g ** 2) for g in grads]) + LMB * sum([torch.sum(p ** 2) for p in list_weights])
            if '2' in mode:
                pen = LAMBDA * sum([n / (d ** 2 + 1e-8) for n, d in zip(noms, denoms)])
            pen.backward()

        D_cost = D_cost_real + D_cost_fake

        torch.nn.utils.clip_grad_norm(netD.parameters(), 0.1)
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

    # if ('dwd' in mode):
    #     list_weights = []
    #     for name, param in netG.named_parameters():
    #         if 'bias' not in name:
    #             list_weights += [param]

    #     grads = autograd.grad(
    #         outputs=G_cost,
    #         inputs=list_weights,
    #         grad_outputs=torch.ones((G_cost).size()).cuda() if use_cuda else torch.ones(
    #             (G_cost).size()),
    #         create_graph=True, retain_graph=True, only_inputs=True)

    #     pen = LAMBDA * sum([torch.sum(g ** 2) for g in grads]) / 2.0
    #     pen.backward()
    torch.nn.utils.clip_grad_norm(netG.parameters(), 0.1)
    optimizerG.step()

    # Write logs and save samples
    # lib.plot.plot('/results/lang_' + mode + '/time', time.time() - start_time)
    # lib.plot.plot('/results/lang_' + mode + '/train disc cost', D_cost.cpu().data.numpy())
    # lib.plot.plot('/results/lang_' + mode + '/train gen cost', G_cost.cpu().data.numpy())

    import ipdb; ipdb.set_trace()

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
        samples = []
        for i in range(10):
            samples.extend(generate_samples(netG))

        with open('/results/lang_' + mode + '/samples_{}.txt'.format(iteration), 'w') as f:
            for s in samples:
                s = "".join(s)
                f.write(s + "\n")

    if iteration % 1000 == 999:
        print('SVDS saved!')
        np.save('/results/lang_' + mode + '/svds', svds)
        svdplot('/results/lang_' + mode + '/')

    if iteration % 100 == 99:
        lib.plot.flush()

    lib.plot.tick()
