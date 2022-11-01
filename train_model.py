from load_ACDC import ACDC_Date
import pylab
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from swin_unet import Swin_Unet
import torch.nn as nn


# 加载数据集
def load_date():
    dataset = ACDC_Date()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=4, sampler=test_sampler)
    return train_loader, test_loader


# 训练
def train_epoch(train_load, test_load):
    epoch_iou = []
    epoch_loss = []
    net.train()
    for image, label in tqdm(train_load):
        image, label = image.to("cuda"), label.to("cuda")
        # label = torch.unsqueeze(label, 1)
        pred = net.forward(image, False)
        loss = loss_fn(pred, label.type(torch.long))
        epoch_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = torch.argmax(pred, dim=1)
            # 计算Iou
            intersection = torch.logical_and(label, pred)
            union = torch.logical_or(label, pred)
            batch_iou = torch.sum(intersection) / torch.sum(union)
            epoch_iou.append(batch_iou.item())

    epoch_test_iou = []
    net.eval()
    with torch.no_grad():
        for image, label in tqdm(test_load):
            image, label = image.to('cuda'), label.to('cuda')
            pred = net(image, False)
            pred = torch.argmax(pred, dim=1)
            # 计算Iou
            intersection = torch.logical_and(label, pred)
            union = torch.logical_or(label, pred)
            batch_iou = torch.sum(intersection) / torch.sum(union)
            epoch_test_iou.append(batch_iou.item())

    return epoch_iou, epoch_test_iou, epoch_loss


if __name__ == '__main__':
    train_data, test_data = load_date()
    net = Swin_Unet(num_classes=4).to("cuda")
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 100
    plt_test_iou = []
    plt_train_iou = []
    for epoch in range(epochs):
        iou, test_iou, loss = train_epoch(train_data, test_data)
        print('Epoch:', epoch,
              'Loss:', round(np.mean(loss), 4),
              'Iou:', round(np.mean(iou), 4),
              'Test_iou:', round(np.mean(test_iou), 4)
              )

        # static_dict = net.state_dict()
        # torch.save(static_dict, './weights/epoch_{},loss_{},Iou_{},test_Iou_{}.pth'
        #            .format(epoch,
        #                    round(np.mean(loss), 4),
        #                    round(np.mean(iou), 4),
        #                    round(np.mean(test_iou), 4)
        #                    ))
        plt_test_iou.append(round(np.mean(test_iou), 4))
        plt_train_iou.append(round(np.mean(iou), 4))

    plt.plot(range(1, epochs + 1), plt_test_iou, plt_train_iou)
    plt.legend(['test_iou', 'train_iou'])
    plt.savefig('./my_figure.png')
    pylab.show()
