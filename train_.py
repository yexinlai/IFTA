#!/usr/bin/python3
# coding=utf-8

import sys
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from lib import dataset
from net import GCPANet
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import logging as logger
from lib.data_prefetcher import DataPrefetcher
from lib.lr_finder import LRFinder
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TAG = "lung"
SAVE_PATH = "your_output_path"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S', filename="train_%s.log" % (TAG), filemode="w")


# Learning rate scheduler
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps * ratio)
    last = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur / total_steps)
    x = np.abs(cur * 2.0 / total_steps - 2.0 * cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr) * cur + min_lr * first - base_lr * total_steps) / (first - total_steps)

    momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1. - x)
    return lr, momentum


BASE_LR = 1e-5
MAX_LR = 0.1
FIND_LR = False  # True


class GCPANet_MaskRCNN(nn.Module):
    def __init__(self, cfg, num_classes):
        super(GCPANet_MaskRCNN, self).__init__()

        # Initialize GCPANet
        self.gcpa_net = GCPANet(cfg)

        # Initialize Mask-RCNN with a ResNet50 backbone
        backbone = torchvision.models.resnet50(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-2])  # 去掉ResNet的最后两个层
        backbone.out_channels = 2048

        # Anchor generator for the FPN, from Mask-RCNN's default configuration
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        # RoI align layer
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        )

        # Initialize Mask-RCNN model
        self.mask_rcnn = MaskRCNN(backbone, num_classes=num_classes,
                                  rpn_anchor_generator=anchor_generator,
                                  box_roi_pool=roi_pooler)

    def forward(self, images, targets=None):
        # Forward pass through GCPANet
        gcpa_out = self.gcpa_net(images)

        # Forward pass through Mask-RCNN
        if self.training:
            losses = self.mask_rcnn(gcpa_out, targets)
            return losses
        else:
            predictions = self.mask_rcnn(gcpa_out)
            return predictions


def train(Dataset, Network, num_classes=91):
    ## dataset
    cfg = Dataset.Config(datapath='yourdata_path', savepath=SAVE_PATH, mode='train', batch=8, lr=0.001,
                         momen=0.9, decay=5e-4, epoch=100)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
    prefetcher = DataPrefetcher(loader)

    ## network
    model = GCPANet_MaskRCNN(cfg, num_classes).to(device)
    model.train(True)

    ## parameter separation
    base, head = [], []
    for name, param in model.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    sw = SummaryWriter(cfg.savepath)
    global_step = 0

    db_size = len(loader)
    if FIND_LR:
        lr_finder = LRFinder(model, optimizer, criterion=None)
        lr_finder.range_test(loader, end_lr=50, num_iter=100, step_mode="exp")
        plt.ion()
        lr_finder.plot()
        import pdb
        pdb.set_trace()

    # Training
    for epoch in range(cfg.epoch):
        prefetcher = DataPrefetcher(loader)
        batch_idx = -1
        image, mask = prefetcher.next()
        while image is not None:
            niter = epoch * db_size + batch_idx
            lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch * db_size, niter, ratio=1.)
            optimizer.param_groups[0]['lr'] = 0.1 * lr  # for backbone
            optimizer.param_groups[1]['lr'] = lr
            optimizer.momentum = momentum
            batch_idx += 1
            global_step += 1

            # Forward pass
            losses = model(image, mask)
            loss = sum(loss_value for loss_value in losses.values())

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log losses
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {k: v.item() for k, v in losses.items()}, global_step=global_step)

            if batch_idx % 10 == 0:
                msg = f"{datetime.datetime.now()} | step: {global_step} | epoch: {epoch+1}/{cfg.epoch} | lr: {optimizer.param_groups[0]['lr']:.6f} | loss: {loss.item():.6f}"
                print(msg)
                logger.info(msg)

            image, mask = prefetcher.next()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.epoch:
            torch.save(model.state_dict(), f"{cfg.savepath}/model-{epoch+1}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(dataset, GCPANet)
