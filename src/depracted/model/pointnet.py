import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def _conv(in_channel, out_channel, kernel_size, padding):
    return nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
                         nn.BatchNorm1d(out_channel),
                         nn.LeakyReLU(negative_slope=0.1, inplace=True))

def _conv3x3(in_channel, out_channel):
    return _conv(in_channel, out_channel, kernel_size=3, padding=1)


def _conv1x1(in_channel, out_channel):
    return _conv(in_channel, out_channel, kernel_size=1, padding=0)

def _fc(in_channel, out_channel, batch_norm=True, nonlinearity=True, dropout=0.0):
    if dropout > 0:
        if batch_norm and nonlinearity:
            return nn.Sequential(nn.Linear(in_channel, out_channel),
                                 nn.Dropout(p=dropout),
                                 nn.BatchNorm1d(out_channel),
                                 nn.LeakyReLU(negative_slope=0.1, inplace=True))
        if batch_norm:
            return nn.Sequential(nn.Linear(in_channel, out_channel),
                                 nn.Dropout(p=dropout),
                                 nn.BatchNorm1d(out_channel))

        if nonlinearity:
            return nn.Sequential(nn.Linear(in_channel, out_channel),
                                 nn.Dropout(p=dropout),
                                 nn.LeakyReLU(negative_slope=0.1, inplace=True))

        return nn.Sequential(nn.Linear(in_channel, out_channel),
                             nn.Dropout(p=dropout))
    else:
        if batch_norm and nonlinearity:
            return nn.Sequential(nn.Linear(in_channel, out_channel),
                                 nn.BatchNorm1d(out_channel),
                                 nn.LeakyReLU(negative_slope=0.1, inplace=True))
        if batch_norm:
            return nn.Sequential(nn.Linear(in_channel, out_channel),
                                 nn.BatchNorm1d(out_channel))

        if nonlinearity:
            return nn.Sequential(nn.Linear(in_channel, out_channel),
                                 nn.LeakyReLU(negative_slope=0.1, inplace=True))

        return nn.Linear(in_channel, out_channel)

def regression_loss(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))

def regression_loss2(pred, target, alpha=0.1):
    # pred[pred[..., -1] > np.pi, -1] -= 2 * np.pi
    # pred[pred[..., -1] < -np.pi, -1] += 2 * np.pi

    loss_l = torch.mean(torch.abs(pred[..., 0] - target[..., 0]))
    loss_w = torch.mean(torch.abs(pred[..., 1] - target[..., 1]))
    loss_ori = torch.mean(torch.abs(pred[..., 2] - target[..., 2]))

    # return (1 - alpha) * (loss_l + loss_w) + alpha * torch.min(torch.Tensor([loss_ori, torch.abs(0.5 * np.pi - loss_ori)]))
    return (loss_l + loss_w) + alpha * loss_ori

# Implementation refers to https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
class TNet(nn.Module):
    def __init__(self, input_dim=3):
        super(TNet, self).__init__()
        self.input_dim = input_dim

        self.conv1 = _conv1x1(input_dim, 64)
        self.conv2 = _conv1x1(64, 128)
        self.conv3 = _conv1x1(128, 1024)
        self.fc1 = _fc(1024, 512)
        self.fc2 = _fc(512, 256)
        self.fc3 = _fc(256, (self.input_dim**2), batch_norm=False, nonlinearity=False)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) #x = (batch_size, 1024, n)

        # a = torch.max(x, 2, keepdim=True)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # # Adding bias
        # identity = Variable(torch.from_numpy(np.eye(self.input_dim).flatten().astype(np.float32))).view(1, self.input_dim**2).repeat(batch_size, 1)
        # if x.is_cuda:
        #     identity = identity.cuda()
        # x = x + identity
        x = x.view(-1, self.input_dim, self.input_dim)
        return x

class PointNet(nn.Module):
    def __init__(self, input_dim=3):
        super(PointNet, self).__init__()
        # self.TNet_input = TNet(input_dim=input_dim)
        # self.TNet_fea = TNet(input_dim=64)
        self.conv1 = _conv1x1(input_dim, 64)
        self.conv2 = _conv1x1(64, 64)
        self.conv3 = _conv1x1(64, 128)
        self.conv4 = _conv1x1(128, 1024)

    def forward(self, x):
        #x = (batch_size, n_channel, n_pts)
        # transform_input = self.TNet_input(x)
        # x = torch.bmm(transform_input, x)
        x = self.conv1(x)
        x = self.conv2(x)
        # transform_fea = self.TNet_fea(x)
        # x = torch.bmm(transform_fea, x)
        x = self.conv3(x)
        x = self.conv4(x) # Point feature (batch_size, n_pts, 1024)
        x = torch.max(x, 2, keepdim=True)[0] # gobal feature
        x = x.view(-1, 1024)

        return x

class BoundingBoxRegressor(PointNet):
    def __init__(self, input_dim=3, target_dim=7):
        super(BoundingBoxRegressor, self).__init__()
        self.pointnet = PointNet(input_dim=input_dim)
        self.fc1 = _fc(1024, 512)
        self.fc2 = _fc(512, 256, dropout=0.3)
        self.fc3 = _fc(256, target_dim, batch_norm=False, nonlinearity=False)

        # self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.L1Loss()
        self.loss_fn = regression_loss2

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pointnet(x.permute(0, 2, 1))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x[..., -1] = torch.tanh(x[..., -1]) # ensure that last channel is within [-1, 1]

        return x



