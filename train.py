import os
import time
import logging
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.ops import sigmoid_focal_loss

import losses
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
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.iou = losses.GIOULoss()

    def forward(self, out, gt):
        # class loss
        gt_cls = gt[:, 5:]
        out_cls = out[:, 5:]
        # loss_cls = self.bce(out_cls, gt_cls)
        N_pos = max(1, torch.sum(gt_cls))
        loss_cls = sigmoid_focal_loss(out_cls, gt_cls, reduction='sum') / N_pos

        # bbox
        gt_reg = gt[:, :4]
        out_reg = out[:, :4]
        loss_reg = self.iou(out_reg, gt_reg)

        # centerness
        gt_cen = gt[:, 4]
        out_cen = out[:, 4]
        loss_cen = self.bce(out_cen, gt_cen)

        loss = loss_cls + loss_reg + loss_cen

        self.loss_cls = loss_cls.detach()
        self.loss_reg = loss_reg.detach()
        self.loss_cen = loss_cen.detach()

        return loss


def train(
    logs_root,
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    batch_size=16,
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

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

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
        transforms.RandomResize(320, 480),
        transforms.RandomCrop(320),
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
                ('epoch', 'time', 'loss T', 'loss V', 'loss_cls', 'loss_reg', 'loss_cen'))

    for epoch in range(epoch_begin, epochs):
        t0 = time.time()
        net.train()
        losses = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

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
        losses_cls = 0
        losses_reg = 0
        losses_cen = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                x = x.to(device)
                y = y.to(device)

                out = net(x)
                loss = criterion(out, y)

                losses += loss.detach()

                losses_cls += criterion.loss_cls
                losses_reg += criterion.loss_reg
                losses_cen += criterion.loss_cen

        loss_val = losses / len(val_loader)
        loss_cls = losses_cls / len(val_loader)
        loss_reg = losses_reg / len(val_loader)
        loss_cen = losses_cen / len(val_loader)
        t1 = time.time()
        time_val = t1 - t0

        time_total = time_train + time_val

        logger.info('| %5d | %8.1f | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f |' %
                    (epoch + 1, time_total, loss_train, loss_val, loss_cls, loss_reg, loss_cen))

        model_file = os.path.join(model_path, 'model_%03d.pt' % (epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss_val.item(),
        }, model_file)


if __name__ == '__main__':
    train(
        logs_root='logs/coco/fcos2',
        epochs=10,
    )
