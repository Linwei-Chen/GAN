__author__ = "charles"
__email__ = "charleschen2013@163.com"
import os
from os import path as osp
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(osp.join(sys.path[0], '../'))
sys.path.append(osp.join(sys.path[0], '../../'))
# sys.path.append(osp.join(sys.path[0], '../../../'))
import time
import torch
import torch.nn as nn
from src.utils.train_utils import model_accelerate, get_device, mean, get_lr
from src.pix2pixHD.train_config import config
from src.pix2pixHD.networks import get_G, get_D, get_E
from torch.optim import Adam
from src.pix2pixHD.hinge_lr_scheduler import get_hinge_scheduler
from src.utils.logger import ModelSaver, Logger
from src.datasets import get_cityscapes_dataloader
from src.pix2pixHD.utils import get_edges, label_to_one_hot, get_encode_features
from src.utils.visualizer import Visualizer
from tqdm import tqdm
from torchvision import transforms
from src.pix2pixHD.criterion import get_GANLoss, get_VGGLoss, get_DFLoss
from tensorboardX import SummaryWriter


def train(args, get_dataloader_func=get_cityscapes_dataloader):
    logger = Logger(save_path=args.save, json_name='seg2img')

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['G', 'D', 'E', 'G_optimizer', 'D_optimizer', 'E_optimizer',
                                        'G_scheduler', 'D_scheduler', 'E_scheduler'])
    visualizer = Visualizer(keys=['image', 'encode_feature', 'fake', 'label', 'instance'])
    sw = SummaryWriter(args.tensorboard_path)
    G = get_G(args)
    D = get_D(args)
    E = get_E(args)
    model_saver.load('G', G)
    model_saver.load('D', D)
    model_saver.load('E', E)

    G_optimizer = Adam(G.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D_optimizer = Adam(D.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))
    E_optimizer = Adam(E.parameters(), lr=args.E_lr, betas=(args.beta1, 0.999))

    model_saver.load('G_optimizer', G_optimizer)
    model_saver.load('D_optimizer', D_optimizer)
    model_saver.load('E_optimizer', E_optimizer)

    G_scheduler = get_hinge_scheduler(args, G_optimizer)
    D_scheduler = get_hinge_scheduler(args, D_optimizer)
    E_scheduler = get_hinge_scheduler(args, E_optimizer)

    model_saver.load('G_scheduler', G_scheduler)
    model_saver.load('D_scheduler', D_scheduler)
    model_saver.load('E_scheduler', E_scheduler)

    device = get_device(args)

    GANLoss = get_GANLoss(args)
    if args.use_ganFeat_loss:
        DFLoss = get_DFLoss(args)
    if args.use_vgg_loss:
        VGGLoss = get_VGGLoss(args)

    epoch_now = len(logger.get_data('G_loss'))
    for epoch in range(epoch_now, args.epochs):
        G_loss_list = []
        D_loss_list = []

        data_loader = get_dataloader_func(args, train=True)
        data_loader = tqdm(data_loader)

        for step, sample in enumerate(data_loader):
            imgs = sample['image'].to(device)
            instances = sample['instance'].to(device)
            labels = sample['label'].to(device)
            smasks = sample['smask'].to(device)
            # print(smasks.shape)

            instances_edge = get_edges(instances)
            one_hot_labels = label_to_one_hot(smasks.long(), n_class=args.label_nc)

            # Encoder out
            encode_features = E(imgs, instances)

            # train the Discriminator
            D_optimizer.zero_grad()
            labels_instE_encodeF = torch.cat([one_hot_labels.float(), instances_edge.float(), encode_features.float()],
                                             dim=1)
            fakes = G(labels_instE_encodeF).detach()

            labels_instE_realimgs = torch.cat([one_hot_labels.float(), instances_edge.float(), imgs.float()], dim=1)
            D_real_outs = D(labels_instE_realimgs)
            D_real_loss = GANLoss(D_real_outs, True)

            labels_instE_fakeimgs = torch.cat([one_hot_labels.float(), instances_edge.float(), fakes.float()], dim=1)
            D_fake_outs = D(labels_instE_fakeimgs)
            D_fake_loss = GANLoss(D_fake_outs, False)

            D_loss = 0.5 * (D_real_loss + D_fake_loss)
            D_loss = D_loss.mean()
            D_loss.backward()
            D_loss = D_loss.item()
            D_optimizer.step()

            # train generator and encoder
            G_optimizer.zero_grad()
            E_optimizer.zero_grad()
            fakes = G(labels_instE_encodeF)
            labels_instE_fakeimgs = torch.cat([one_hot_labels.float(), instances_edge.float(), fakes.float()], dim=1)
            D_fake_outs = D(labels_instE_fakeimgs)

            gan_loss = GANLoss(D_fake_outs, True)

            G_loss = 0
            G_loss += gan_loss
            gan_loss = gan_loss.mean().item()

            if args.use_vgg_loss:
                vgg_loss = VGGLoss(fakes, imgs)
                G_loss += args.lambda_feat * vgg_loss
                vgg_loss = vgg_loss.mean().item()
            else:
                vgg_loss = 0.

            if args.use_ganFeat_loss:
                df_loss = DFLoss(D_fake_outs, D_real_outs)
                G_loss += args.lambda_feat * df_loss
                df_loss = df_loss.mean().item()
            else:
                df_loss = 0.

            G_loss = G_loss.mean()
            G_loss.backward()
            G_loss = G_loss.item()

            G_optimizer.step()
            E_optimizer.step()

            data_loader.write(f'Epochs:{epoch} | Dloss:{D_loss:.6f} | Gloss:{G_loss:.6f}'
                              f'| GANloss:{gan_loss:.6f} | VGGloss:{vgg_loss:.6f} '
                              f'| DFloss:{df_loss:.6f} | lr:{get_lr(G_optimizer):.8f}')

            G_loss_list.append(G_loss)
            D_loss_list.append(D_loss)

            # display
            if args.display and step % args.display == 0:
                visualizer.display(transforms.ToPILImage()(encode_features[0].cpu()), 'encode_feature')
                visualizer.display(transforms.ToPILImage()(imgs[0].cpu()), 'image')
                visualizer.display(transforms.ToPILImage()(fakes[0].cpu()), 'fake')
                visualizer.display(transforms.ToPILImage()(labels[0].cpu() * 15), 'label')
                visualizer.display(transforms.ToPILImage()(instances[0].cpu() * 15), 'instance')

            # tensorboard log
            if args.tensorboard_log and step % args.tensorboard_log == 0:
                total_steps = epoch * len(data_loader) + step
                sw.add_scalar('Loss/G', G_loss, total_steps)
                sw.add_scalar('Loss/D', D_loss, total_steps)
                sw.add_scalar('Loss/gan', gan_loss, total_steps)
                sw.add_scalar('Loss/vgg', vgg_loss, total_steps)
                sw.add_scalar('Loss/df', df_loss, total_steps)

                sw.add_scalar('LR/G', get_lr(G_optimizer), total_steps)
                sw.add_scalar('LR/D', get_lr(D_optimizer), total_steps)
                sw.add_scalar('LR/E', get_lr(E_optimizer), total_steps)

                sw.add_image('img/real', imgs[0].cpu(), step)
                sw.add_image('img/fake', fakes[0].cpu(), step)
                sw.add_image('visual/encode_feature', encode_features[0].cpu(), step)
                sw.add_image('visual/instance', instances[0].cpu(), step)
                sw.add_image('visual/label', labels[0].cpu(), step)

        D_scheduler.step(epoch)
        G_scheduler.step(epoch)
        E_scheduler.step(epoch)

        logger.log(key='D_loss', data=sum(D_loss_list) / float(len(D_loss_list)))
        logger.log(key='G_loss', data=sum(G_loss_list) / float(len(G_loss_list)))
        logger.save_log()
        logger.visualize()

        model_saver.save('G', G)
        model_saver.save('D', D)
        model_saver.save('E', E)

        model_saver.save('G_optimizer', G_optimizer)
        model_saver.save('D_optimizer', D_optimizer)
        model_saver.save('E_optimizer', E_optimizer)

        model_saver.save('G_scheduler', G_scheduler)
        model_saver.save('D_scheduler', D_scheduler)
        model_saver.save('E_scheduler', E_scheduler)


if __name__ == '__main__':
    args = config()
    train(args, get_dataloader_func=get_cityscapes_dataloader)

pass
