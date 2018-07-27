import os, sys
sys.path.append(os.getcwd())

import time
# import tflib as lib
# import tflib.save_images
# import tflib.mnist
import cifar10
# import tflib.plot
# import tflib.inception_score

import numpy as np
from inception_score import get_inception_score
from itertools import chain
import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torch.optim.optimizer import Optimizer
import math

class Adamp(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, md=1):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.md = md
        super(Adamp, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adamp does not support sparse gradients, please consider SparseAdamp instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                if self.md == 0:
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                else:
                    exp_avg.mul_(beta1).add_(1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss



DATA_DIR = '/mnt/dataset1/cifar-10-batches-py'

# MODE = 'wgan-gp' # dcgan, dcgan-nm, dcgan-nm-sat
MODE = str(sys.argv[1])
DIM = 128
LAMBDA = 10
if MODE == 'wgan-gp':
    CRITIC_ITERS = 5
else:
    CRITIC_ITERS = 1
BATCH_SIZE = 64
ITERS = 200000
OUTPUT_DIM = 3072

lfile = open("/results/inception_score_" + MODE + ".txt", "w")


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
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
print(netG)
print(netD)

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

if 'nm' in MODE:
    print('NMMMMMMMMMMMMM')
    optimizerD = Adamp(netD.parameters(), lr=1e-4, betas=(-0.5, 0.9))
else:
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

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
# def generate_image(frame, netG):
#     fixed_noise_128 = torch.randn(128, 128)
#     if use_cuda:
#         fixed_noise_128 = fixed_noise_128.cuda(gpu)
#     noisev = autograd.Variable(fixed_noise_128, volatile=True)
#     samples = netG(noisev)
#     samples = samples.view(-1, 3, 32, 32)
#     samples = samples.mul(0.5).add(0.5)
#     samples = samples.cpu().data.numpy()

#     lib.save_images.save_images(samples, './tmp/cifar10/samples_{}.jpg'.format(frame))

# For calculating inception score
# def get_inception_score(G, ):
#     all_samples = []
#     for i in xrange(10):
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
train_gen, dev_gen = cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)
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

criterion = nn.BCEWithLogitsLoss()
criterion.cuda()

for iteration in range(ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for i in range(CRITIC_ITERS):
        _data = gen.__next__()
        netD.zero_grad()

        # train with real
        _data = _data.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
        real_data = torch.stack([preprocess(item) for item in _data])

        if use_cuda:
            real_data = real_data.cuda(gpu)
        real_data_v = autograd.Variable(real_data)

        # import torchvision
        # filename = os.path.join("test_train_data", str(iteration) + str(i) + ".jpg")
        # torchvision.utils.save_image(real_data, filename)

        pred_real = netD(real_data_v)
        if MODE == 'wgan-gp':
            pred_real = pred_real.mean()
            pred_real.backward(mone)
        else:
            label = torch.ones(BATCH_SIZE, 1)
            label = autograd.Variable(label).cuda(gpu)
            D_cost = criterion(pred_real, label)
            D_cost.backward()

        # train with fake
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        fake = autograd.Variable(netG(noisev).data)
        inputv = fake
        pred_fake = netD(inputv)
        if MODE == 'wgan-gp':
            pred_fake = pred_fake.mean()
            pred_fake.backward(one)
        else:
            label = torch.zeros(BATCH_SIZE, 1)
            label = autograd.Variable(label).cuda(gpu)
            D_cost = criterion(pred_fake, label)
            D_cost.backward()

        # train with gradient penalty
        if MODE == 'wgan-gp':
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
            gradient_penalty.backward()

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
    pred_fake = netD(fake)
    if MODE == 'wgan-gp':
        pred_fake = pred_fake.mean()
        pred_fake.backward(mone)
    else:
        if 'sat' in MODE:
            label = torch.zeros(BATCH_SIZE, 1)
            label = autograd.Variable(label).cuda(gpu)
            G_cost = -criterion(pred_fake, label)
            G_cost.backward()
        else:
            label = torch.ones(BATCH_SIZE, 1)
            label = autograd.Variable(label).cuda(gpu)
            G_cost = criterion(pred_fake, label)
            G_cost.backward()
    optimizerG.step()

    # Write logs and save samples
    # lib.plot.plot('./tmp/cifar10/train disc cost', D_cost.cpu().data.numpy())
    # lib.plot.plot('./tmp/cifar10/time', time.time() - start_time)
    # lib.plot.plot('./tmp/cifar10/train gen cost', G_cost.cpu().data.numpy())
    # lib.plot.plot('./tmp/cifar10/wasserstein distance', Wasserstein_D.cpu().data.numpy())

    # Calculate inception score every 1K iters
    # if False and iteration % 1000 == 999:
    #     inception_score = get_inception_score(netG)
        # lib.plot.plot('./tmp/cifar10/inception score', inception_score[0])

    # Calculate dev loss and generate samples every 100 iters
    if iteration % 200 == 1:
        sample_list = []
        for i in range(20):
            print(i)
            z = autograd.Variable(torch.randn(200, 128)).cuda(gpu)
            samples = netG(z)
            sample_list.append(samples.data.cpu().numpy())

        # Flattening list of lists into one list of numpy arrays
        new_sample_list = list(chain.from_iterable(sample_list))
        print("Calculating Inception Score over 8k generated images")
        # Feeding list of numpy arrays
        inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                                              resize=True, splits=10)

        print("Inception score: {}".format(inception_score))
        print("Generator iter: {}".format(iteration))

        output = str(iteration) + " " + str(inception_score[0]) + "\n"
        lfile.write(output)

        # dev_disc_costs = []
        # for images in dev_gen():
        #     images = images.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
        #     imgs = torch.stack([preprocess(item) for item in images])

        #     # imgs = preprocess(images)
        #     if use_cuda:
        #         imgs = imgs.cuda(gpu)
        #     imgs_v = autograd.Variable(imgs, volatile=True)

        #     D = netD(imgs_v)
        #     _dev_disc_cost = -D.mean().cpu().data.numpy()
        #     dev_disc_costs.append(_dev_disc_cost)
        # lib.plot.plot('./tmp/cifar10/dev disc cost', np.mean(dev_disc_costs))

        # generate_image(iteration, netG)

    # Save logs every 100 iters
    # if (iteration < 5) or (iteration % 100 == 99):
    #     lib.plot.flush()
    # lib.plot.tick()
