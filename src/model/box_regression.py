import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .box_regression_fn import _model_fn, _model_eval_fn

_INPUT_WITH_ANGLE = True


def _conv(in_channel, out_channel, kernel_size, padding):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm1d(out_channel),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )


def _conv3x3(in_channel, out_channel):
    return _conv(in_channel, out_channel, kernel_size=3, padding=1)


def _conv1x1(in_channel, out_channel):
    return _conv(in_channel, out_channel, kernel_size=1, padding=0)


def _fc(in_channel, out_channel, batch_norm=True, nonlinearity=True):

    if batch_norm and nonlinearity:
        return nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
    if batch_norm:
        return nn.Sequential(
            nn.Linear(in_channel, out_channel), nn.BatchNorm1d(out_channel)
        )

    if nonlinearity:
        return nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    return nn.Linear(in_channel, out_channel)


def regression_loss(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))


def regression_loss2(pred, target, alpha=0.5):
    if (
        pred.shape[1] == 5
    ):  # 3d box regression, with z coordinate being regressed as well
        loss_z = torch.mean(torch.abs(pred[..., 0] - target[..., 0]))
        loss_dim = torch.mean(
            torch.sum(torch.abs(pred[:, 1:-1] - target[:, 1:-1]), dim=1)
        )
        loss_ori = torch.mean(torch.abs(pred[..., -1] - target[..., -1]))
        return loss_z + loss_dim + alpha * loss_ori
    elif pred.shape[1] == 3:  # 3d box regression
        loss_dim = torch.mean(
            torch.sum(torch.abs(pred[:, :-1] - target[:, :-1]), dim=1)
        )
        loss_ori = torch.mean(torch.abs(pred[..., -1] - target[..., -1]))
        return loss_dim + alpha * loss_ori


# Implementation based on https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
class TNet(nn.Module):
    def __init__(self, input_dim=3):
        super(TNet, self).__init__()
        self.input_dim = input_dim

        self.conv1 = _conv1x1(input_dim, 64)
        self.conv2 = _conv1x1(64, 128)
        self.conv3 = _conv1x1(128, 1024)
        self.fc1 = _fc(1024, 512)
        self.fc2 = _fc(512, 256)
        self.fc3 = _fc(256, (self.input_dim ** 2), batch_norm=False, nonlinearity=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # x = (batch_size, 1024, n)

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
        self.conv1 = _conv1x1(input_dim, 64)
        self.conv2 = _conv1x1(64, 64)
        self.conv3 = _conv1x1(64, 128)
        self.conv4 = _conv1x1(128, 1024)

    def forward(self, x):
        # x = (batch_size, n_channel, n_pts)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # Point feature (batch_size, n_pts, 1024)
        x = torch.max(x, 2, keepdim=True)[0]  # gobal feature
        x = x.view(-1, 1024)

        return x


class BoundingBoxRegressor(PointNet):
    def __init__(self, cfg):
        super(BoundingBoxRegressor, self).__init__()
        self.dropout = cfg["dropout"]

        self.backbone = PointNet(input_dim=cfg["input_dim"])
        self.fc1 = _fc(1024, 512)
        self.fc2 = _fc(512, 256)
        self.fc3 = _fc(256, cfg["target_dim"], batch_norm=False, nonlinearity=False)

        # self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.L1Loss()
        self.loss_fn = regression_loss2

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def model_fn(model, batch_data):
        return _model_fn(model, batch_data)

    @staticmethod
    def model_eval_fn(model, batch_data):
        return _model_eval_fn(model, batch_data)

    # @staticmethod
    # def model_eval_collate_fn(tb_dict_list, eval_dict_list):
    #     return _model_eval_collate_fn(tb_dict_list, eval_dict_list)

    def forward(self, x):
        x = self.backbone(x.permute(0, 2, 1))
        x = self.fc1(x)
        x = self.fc2(x)

        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc3(x)
        # x[..., -1] = torch.tanh(x[..., -1]) # ensure that last channel is within [-1, 1]

        return x
