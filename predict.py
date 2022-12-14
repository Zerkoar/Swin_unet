import pylab
from matplotlib import pyplot as plt

from swin_unet import Swin_Unet
from load_ACDC import ACDC_Date
from load_LIDC_data import LIDC_IDRI
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from unet import Unet

# dataset = LIDC_IDRI(dataset_location='./Data/')
dataset = ACDC_Date()
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler, pin_memory=True)
test_loader = DataLoader(dataset, batch_size=8, sampler=test_sampler)
Iou = []


def iou_mean(pred, target, n_classes):
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


def Predict_result():
    model = Swin_Unet(num_classes=4).to("cuda")
    model.load_state_dict(torch.load('./ACDC_weights/SwinUnet_epoch_188,loss_0.0061,Iou_0.7267,test_Iou_0.5995.pth'))
    model.eval()

    img, label = next(iter(test_loader))
    img = img.to('cuda')
    label = label.to('cuda')
    label = torch.unsqueeze(label, 1)
    pred = model.forward(img)
    pred = torch.argmax(pred, dim=1)
    ious = iou_mean(pred, label, 4)
    print(ious)
    pred = torch.unsqueeze(pred, dim=1)
    plt.figure(figsize=(25, 8))

    column = 3
    for i in range(pred.shape[0]):
        plt.subplot(column, 15, i + 1)
        plt.imshow(img[i].permute(1, 2, 0).cpu().numpy())
        plt.subplot(column, 15, i + 16)
        # plt.title("label")
        plt.imshow(label[i].permute(1, 2, 0).cpu().numpy())
        plt.subplot(column, 15, i + 31)
        # plt.title("predict")
        plt.imshow(pred[i].permute(1, 2, 0).cpu().detach().numpy())
    pylab.show()


if __name__ == '__main__':
    Predict_result()
