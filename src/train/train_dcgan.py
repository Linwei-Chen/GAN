__author__ = "charles"
__email__ = "charleschen2013@163.com"
from os import path as osp
import sys

sys.path.append(osp.join(sys.path[0], '../'))
sys.path.append(osp.join(sys.path[0], '../../'))
import time
import torch
import torch.nn as nn
import torch.optim as optim
from model.cDCGAN import Generator, Discriminator
from utils.train_config import config
from utils.logger import Logger, ModelSaver
from utils.train_utils import model_accelerate, get_device, mean
from datasets import get_cifar_10_data_set, get_mnist_data_set
from utils.visualizer import Visualizer
from utils.train_utils import get_lr
from tqdm import tqdm

args = config()

# fixed noise & label
temp_z_ = torch.randn(10, 100)
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(10, 1)
for i in range(9):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(10, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)

fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
fixed_y_label_ = torch.zeros(100, 10)
fixed_y_label_.scatter_(1, fixed_y_.long(), 1)
fixed_y_label_ = fixed_y_label_.view(-1, 10, 1, 1)

logger = Logger(save_path=args.save, json_name='record')
model_saver = ModelSaver(save_path=args.save, name_list=[
    'D',
    'G',
    f'D_{args.optimizer}',
    f'G_{args.optimizer}',
    f'D_{args.scheduler}',
    f'G_{args.scheduler}'
])

# get model and accelerate the model training

# network
G = Generator(128)
D = Discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)

G = model_accelerate(args, G)
model_saver.load(name='G', model=G)
D = model_accelerate(args, D)
model_saver.load(name='D', model=D)

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
model_saver.load(name=f'G_{args.optimizer}', model=G_optimizer)
D_optimizer = optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
model_saver.load(name=f'D_{args.optimizer}', model=D_optimizer)

G_scheduler = optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=args.milestones, gamma=0.1)
model_saver.load(name=f'G_{args.scheduler}', model=G_scheduler)
D_scheduler = optim.lr_scheduler.MultiStepLR(D_optimizer, milestones=args.milestones, gamma=0.1)
model_saver.load(name=f'D_{args.scheduler}', model=D_scheduler)

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# label preprocess
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).long().view(10, 1), 1).view(10, 10, 1, 1)
fill = torch.zeros([10, 10, args.crop_size, args.crop_size])
for i in range(10):
    fill[i, i, :, :] = 1

print('training start!')
device = get_device(args)
epoch_start = len(logger.get_data('G_train_loss'))
visualizer = Visualizer(keys=['fake', 'real'])
for epoch in range(epoch_start, args.epochs):
    G_scheduler.step()
    D_scheduler.step()
    y_real_ = torch.ones(args.batch_size).to(device)
    y_fake_ = torch.zeros(args.batch_size).to(device)
    if args.dcgan == 'mnist':
        train_loader = get_mnist_data_set(args)
    elif args.dcgan == 'cifar10':
        train_loader = get_cifar_10_data_set(args)
    else:
        raise NotImplementedError
    train_loader = tqdm(train_loader)

    D_fake_loss_list = []
    D_fake_score_list = []
    G_train_loss_list = []
    iter = 0
    for (x_, y_) in train_loader:
        iter += 1
        # train discriminator D
        t1 = time.perf_counter()
        D.zero_grad()
        y_fill_ = fill[y_].to(device)
        D_result = D(x_, y_fill_).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((args.batch_size, 100)).view(-1, 100, 1, 1).to(device)
        y_ = (torch.rand(args.batch_size, 1) * 10).long().squeeze()
        y_label_ = onehot[y_].to(device)
        y_fill_ = fill[y_].to(device)

        G_result = G(z_, y_label_)
        D_result = D(G_result, y_fill_).squeeze()

        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result.mean().item()

        D_fake_loss_list.append(D_fake_loss)
        D_fake_score_list.append(D_fake_score)

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        # train generator G
        G.zero_grad()
        z_ = torch.randn((args.batch_size, 100)).view(-1, 100, 1, 1).to(device)
        y_ = (torch.rand(args.batch_size, 1) * 10).long().squeeze()
        y_label_ = onehot[y_].to(device)
        y_fill_ = fill[y_].to(device)

        G_result = G(z_, y_label_)
        D_result = D(G_result, y_fill_).squeeze()

        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss_list.append(G_train_loss)

        G_train_loss.backward()
        G_optimizer.step()
        t2 = time.perf_counter()
        train_loader.write(f'Epoch:{epoch} '
                           f'| {D_fake_loss} | {D_fake_score} | {G_train_loss} | time:{t2 - t1} '
                           f'| lr:{get_lr(G_optimizer)}')

        if args.display and iter % args.display_iter == 0:
            visualizer.display(x_[0].cpu(), 'real')
            visualizer.display(G_result[0].detach().cpu().numpy(), 'fake')

    model_saver.save(name='D', model=D)
    model_saver.save(name='G', model=G)
    model_saver.save(name=f'D_{args.optimizer}', model=D_optimizer)
    model_saver.save(name=f'G_{args.optimizer}', model=G_optimizer)
    model_saver.save(name=f'D_{args.scheduler}', model=D_scheduler)
    model_saver.save(name=f'G_{args.scheduler}', model=G_scheduler)

    logger.log(key='D_real_loss', data=float(mean(D_fake_loss_list)), show=True)
    logger.log(key='D_fake_loss', data=float(mean(D_fake_score_list)), show=True)
    logger.log(key='G_train_loss', data=float(mean(G_train_loss_list)), show=True)
    logger.save_log()
    logger.visualize()
