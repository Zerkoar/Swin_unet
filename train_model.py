import random
import time

from load_ACDC import ACDC_Date
# from load_LIDC_data import LIDC_IDRI
import pylab
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from swin_unet import Swin_Unet
import torch.nn as nn

seed = 3
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# 加载数据集
def load_date():
    dataset = ACDC_Date()
    # dataset = LIDC_IDRI(dataset_location='./Data/')
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
    acc = []
    net.train()
    time.sleep(0.1)
    for image, label in tqdm(train_load):
        image, label = image.to("cuda"), label.to("cuda")
        pred = net.forward(image)
        loss = loss_fn(pred, label.type(torch.long))
        epoch_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            acc.append(multi_acc(pred, label).item())
            pred = torch.argmax(pred, dim=1)
            # 计算Iou
            epoch_iou.append(iou_mean(pred, label))

    epoch_test_iou = []
    test_acc = []
    net.eval()
    with torch.no_grad():
        for image, label in tqdm(test_load):
            image, label = image.to('cuda'), label.to('cuda')
            pred = net(image)
            test_acc.append(multi_acc(pred, label).item())
            pred = torch.argmax(pred, dim=1)
            # 计算Iou
            epoch_test_iou.append(iou_mean(pred, label))

    return epoch_iou, epoch_test_iou, epoch_loss, acc, test_acc


def iou_mean(pred, target, n_classes=4):
    # n_classes ：the number of classes in your dataset,not including background
    # for mask and ground-truth label, not probability map
    ious = []
    iousSum = 0
    pred = pred.view(-1)
    target = np.array(target.cpu())
    target = torch.from_numpy(target)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
    return iousSum / n_classes


def multi_acc(pred, label):
    probs = torch.log_softmax(pred.type(torch.float32), dim=1)
    _, tags = torch.max(probs, dim=1)
    corrects = torch.eq(tags, label).int()
    acc = corrects.sum() / corrects.numel()
    return acc


if __name__ == '__main__':
    train_data, test_data = load_date()
    net = Swin_Unet(num_classes=4).to("cuda")
    # net.load_state_dict(torch.load('./ACDC_weights/acc_epoch_198,loss_0.0657,Iou_0.5381,test_Iou_0.4672.pth'))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 200
    plt_test_iou = []
    plt_train_iou = []
    for epoch in range(epochs):
        iou, test_iou, loss, acc, test_acc = train_epoch(train_data, test_data)
        print('Epoch:', epoch,
              'Loss:', round(np.mean(loss), 4),
              'Iou:', round(np.mean(iou), 4),
              'Test_iou:', round(np.mean(test_iou), 4),
              'Acc:', round(np.mean(acc), 4),
              'Test_acc:', round(np.mean(test_acc), 4)
              )

        if round(np.mean(test_iou), 4) > 0.5 or epoch > 198:
            static_dict = net.state_dict()
            torch.save(static_dict, './ACDC_weights/SwinUnet_epoch_{},loss_{},Iou_{},test_Iou_{}.pth'
                       .format(epoch,
                               round(np.mean(loss), 4),
                               round(np.mean(iou), 4),
                               round(np.mean(test_iou), 4)
                               ))

        plt_test_iou.append(round(np.mean(test_iou), 4))
        plt_train_iou.append(round(np.mean(iou), 4))

    plt.plot(range(1, epochs + 1), plt_test_iou, plt_train_iou)
    plt.legend(['test_iou', 'train_iou'])
    plt.savefig('./my_figure.png')
    pylab.show()
