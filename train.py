import os
import time
import logging
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import models
import datasets
import transforms


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class Loss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1loss = nn.L1Loss(reduction='sum')

    def forward(self, out, gt):

        out_k = out[:, 4:]
        out_ox = out[:, 0]
        out_oy = out[:, 1]
        out_sx = out[:, 2]
        out_sy = out[:, 3]

        gt_k = gt[:, 4:]
        gt_ox = gt[:, 0]
        gt_oy = gt[:, 1]
        gt_sx = gt[:, 2]
        gt_sy = gt[:, 3]

        positive = gt_k == 1
        negative = positive == False

        out_k = torch.sigmoid(out_k)
        out_k_pos = out_k[positive]
        out_k_neg = out_k[negative]
        gt_k_neg = gt_k[negative]

        N = len(out_k_pos)

        Lk_pos = torch.sum((1 - out_k_pos)**self.alpha * torch.log(out_k_pos))
        Lk_neg = torch.sum((1 - gt_k_neg)**self.beta *
                           out_k_neg ** self.alpha * torch.log(1 - out_k_neg))
        Lk = -(Lk_pos + Lk_neg) / (N + 1e-6)

        positive_ = gt_sx > 0

        out_ox_pos = out_ox[positive_]
        out_oy_pos = out_oy[positive_]
        gt_ox_pos = gt_ox[positive_]
        gt_oy_pos = gt_oy[positive_]

        Lo = (self.l1loss(out_ox_pos, gt_ox_pos) +
              self.l1loss(out_oy_pos, gt_oy_pos)) / (N + 1e-6)

        out_sx_pos = out_sx[positive_]
        out_sy_pos = out_sy[positive_]
        gt_sx_pos = gt_sx[positive_]
        gt_sy_pos = gt_sy[positive_]

        Ls = (self.l1loss(out_sx_pos, gt_sx_pos) +
              self.l1loss(out_sy_pos, gt_sy_pos)) / (N + 1e-6)
        Ls *= 0.1

        loss = Lk + Lo + Ls

        self.loss_k = Lk.detach()
        self.loss_o = Lo.detach()
        self.loss_s = Ls.detach()

        return loss


def train(
    logs_root,
    learning_rate=0.0003,
    weight_decay=0,
    batch_size=8,
    epochs=5,
    num_workers=2,
):

    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(
        os.path.join(logs_root, 'train.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = models.CenterNet()
    net = net.to(device)

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    epoch_begin = 0
    model_files = sorted(os.listdir(model_path))
    if len(model_files):
        checkpoint_path = os.path.join(model_path, model_files[-1])
        print('Load Checkpoint -> ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch']

    T_train = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    T_val = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    root = '../data/coco/train2017'
    annFile = '../data/coco/annotations/instances_train2017.json'
    train_dataset = datasets.CocoDetection(root, annFile, T_train)
    root = '../data/coco/val2017'
    annFile = '../data/coco/annotations/instances_val2017.json'
    val_dataset = datasets.CocoDetection(root, annFile, T_val)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    criterion = Loss()

    # logger.info(net)
    # logger.info(device)
    logger.info(optimizer)

    logger.info('| %5s | %8s | %8s | %8s | %8s | %8s | %8s |' %
                ('epoch', 'time', 'loss T', 'loss V', 'loss_k', 'loss_o', 'loss_s'))

    for epoch in range(epoch_begin, epochs):
        t0 = time.time()
        net.train()
        losses = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = net(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            losses += loss.detach()

        loss_train = losses / len(train_loader)
        t1 = time.time()
        time_train = t1 - t0

        t0 = time.time()
        net.eval()
        losses = 0
        losses_k = 0
        losses_o = 0
        losses_s = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                x = x.to(device)
                y = y.to(device)

                out = net(x)
                loss = criterion(out, y)

                losses += loss.detach()

                losses_k += criterion.loss_k
                losses_o += criterion.loss_o
                losses_s += criterion.loss_s

        loss_val = losses / len(val_loader)
        loss_k = losses_k / len(val_loader)
        loss_o = losses_o / len(val_loader)
        loss_s = losses_s / len(val_loader)
        t1 = time.time()
        time_val = t1 - t0

        time_total = time_train + time_val

        logger.info('| %5d | %8.1f | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f |' %
                    (epoch + 1, time_total, loss_train, loss_val, loss_k, loss_o, loss_s))

        model_file = os.path.join(model_path, 'model_%03d.pt' % (epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss_val.item(),
        }, model_file)


if __name__ == '__main__':
    train(
        logs_root='logs/coco/test',
        epochs=5
    )
