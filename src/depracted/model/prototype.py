import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _encode_decode(in_channel, out_channel, stride=1):
    if in_channel != out_channel or stride != 1:
        block = nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
                              nn.BatchNorm1d(out_channel),
                              nn.LeakyReLU(negative_slope=0.01, inplace=True))
    else:
        block = nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
                              nn.BatchNorm1d(out_channel),
                              nn.LeakyReLU(negative_slope=0.01, inplace=True))
    return block

def test_conv(in_channel, out_channel):
    return nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1),
                         nn.BatchNorm1d(out_channel),
                         nn.LeakyReLU(negative_slope=0.01, inplace=True))

def _pw_conv(in_channel, out_channel):
    return nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size=1),
                         nn.BatchNorm1d(out_channel),
                         nn.LeakyReLU(negative_slope=0.01, inplace=True))

def flow_loss(pred, target, mask=None):
    err_batch = torch.mean(torch.norm(pred - target, dim=-1), dim=1)
    loss = torch.mean(err_batch)
    # loss = torch.mean(torch.norm(pred - target, dim=-1))

    return loss, err_batch

class Prototype(nn.Module):
    def __init__(self, in_channel=1, max_displacement=5): #Currently in_channel = 1 only for debug. n_channel will be 2 if we use xy coords for scan
        super(Prototype, self).__init__()
        #Only structure. Parameters are meaning less currently
        self.max_displacement = max_displacement

        self.encoder_0 = _encode_decode(in_channel, 64, 2)
        self.encoder_1 = _encode_decode(64, 128, 2)
        self.encoder_2 = _encode_decode(128, 256, 2)
        self.decoder_1 = _encode_decode(2 * self.max_displacement + 1 + 128, 128)
        self.decoder_0 = _encode_decode(128 + 64, 128)

        self.flow_reg = _pw_conv(128 + in_channel, 2)

        self.loss_fn = flow_loss

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, scan1, scan2=None):
        batch_size, n_pts, n_channel = scan1.shape
        # scan1 = scan1.reshape(batch_size, -1, n_pts)

        if scan2 is None:
            scan2 = scan1

        scan1 = scan1.permute(0, 2, 1)
        scan2 = scan2.permute(0, 2, 1)

        #Encoding
        # shape: (batch_size, n_channel, n_pts
        feat1_down0 = self.encoder_0(scan1)
        feat2_down0 = self.encoder_0(scan2)
        shape1 = feat1_down0.shape #(5, 64, 225)

        feat1_down1 = self.encoder_1(feat1_down0)
        feat2_down1 = self.encoder_1(feat2_down0)
        shape2 = feat1_down1.shape #(5, 128, 113)

        feat1_down2 = self.encoder_2(feat1_down1)
        feat2_down2 = self.encoder_2(feat2_down1)
        shape3 = feat1_down2.shape #(5, 256, 57)

        #Feature fusion
        feat = self._fusion(feat1_down2, feat2_down2, max_displacement=self.max_displacement)
        shape4 = feat.shape #(5, 256, 57)

        #Decoding
        up1 = self._upsample(feat, size=feat1_down1.shape[-1])
        up1 = torch.cat((feat1_down1, up1), dim=1)
        up1 = self.decoder_1(up1)
        shape5 = up1.shape #(5, 128, 113)

        up0 = self._upsample(up1, size=feat1_down0.shape[-1])
        c = up0.shape
        d = feat1_down0.shape
        up0 = torch.cat((feat1_down0, up0), dim=1)
        e = up0.shape
        up0 = self.decoder_0(up0)
        shape6 = up0.shape #(5, 256, 225)

        out = self._upsample(up0, size=scan1.shape[-1])
        a = out.shape
        b = scan1.shape
        out = torch.cat((scan1, out), dim=1)
        out = self.flow_reg(out)
        shape7 = out.shape

        # return out.reshape(batch_size, n_pts, -1)
        out = out.permute(0, 2, 1)

        return out


    def _upsample(self, x, size):
        return F.interpolate(x, size=size, mode='nearest')

    def _fusion_test(self, feat1, feat2):
        return feat1 + feat2

    def _fusion(self, feat1, feat2, kernel_size=3, max_displacement=5):
        batch_size, n_channel, n_pts = feat1.shape
        half_kernel = kernel_size // 2
        feat_fused = torch.zeros((batch_size, 2 * max_displacement + 1, n_pts)).cuda()

        # indices of patches of scan1
        patch_ids = torch.arange(n_pts).unsqueeze(dim=-1).long()
        window_ids = torch.arange(-half_kernel, half_kernel + 1).long()
        patch_ids = patch_ids + window_ids.unsqueeze(dim=0)
        patch_ids = patch_ids.clamp(min=0, max=n_pts - 1)

        patch1 = feat1[:, :, patch_ids.reshape(-1)].reshape(batch_size, n_channel, n_pts, kernel_size)
        patch1 = patch1.permute(0, 1, 3, 2) #Permute the shape into (batch_size, n_channel, kernel_size, n_pts)
        patch1 = patch1.reshape(batch_size, n_channel * kernel_size, n_pts)

        patch2 = feat2[:, :, patch_ids.reshape(-1)].reshape(batch_size, n_channel, n_pts, kernel_size)
        patch2 = patch2.permute(0, 1, 3, 2)
        patch2 = patch2.reshape(batch_size, n_channel * kernel_size, n_pts)

        patch_corr = torch.matmul(patch1.permute(0, 2, 1), patch2) # correlation matrix for each pair of patches

        # indices of points within max displacement
        patch2_ids = torch.arange(n_pts).unsqueeze(dim=-1).long()
        displacement_ids = torch.arange(-max_displacement, max_displacement + 1).long()
        patch2_ids = patch2_ids + displacement_ids.unsqueeze(dim=0)
        patch2_ids = patch2_ids.clamp(min=0, max=n_pts - 1)
        patch1_ids = torch.arange(n_pts).unsqueeze(dim=-1).expand_as(patch2_ids).long()
        patch_corr_ids = torch.stack((patch1_ids, patch2_ids), dim=2).reshape(-1, 2)

        # mask = torch.zeros(n_pts, n_pts).float().cuda()
        # mask[patch_corr_ids[:, 0], patch_corr_ids[:, 1]] = 1.0
        # patch_corr = patch_corr * mask

        corr_max_displacement = patch_corr[:, patch_corr_ids[:, 0], patch_corr_ids[:, 1]].reshape(batch_size, n_pts, -1)
        feat_fused = corr_max_displacement.permute(0, 2, 1)

        a = feat_fused / feat_fused[0, 0]

        return feat_fused

class PrototypeTest(nn.Module):
    def __init__(self, in_channel=1, max_displacement=5): #Currently in_channel = 1 only for debug. n_channel will be 2 if we use xy coords for scan
        super(PrototypeTest, self).__init__()
        #Only structure. Parameters are meaning less currently
        self.max_displacement = max_displacement

        self.conv1 = test_conv(in_channel, 32)
        self.conv2 = test_conv(32, 64)
        self.conv3 = test_conv(128, 64)
        self.conv4 = test_conv(64, 32)
        self.flow_reg = _pw_conv(32, 2)

        self.loss_fn = flow_loss

    def forward(self, scan1, scan2=None):
        batch_size, n_pts, n_channel = scan1.shape
        # scan1 = scan1.reshape(batch_size, -1, n_pts)

        if scan2 is None:
            scan2 = scan1

        scan1 = scan1.permute(0, 2, 1)
        scan2 = scan2.permute(0, 2, 1)

        feat1 = self.conv1(scan1)
        feat1 = self.conv2(feat1)
        feat2 = self.conv1(scan2)
        feat2 = self.conv2(feat2)
        feat = self._fusion_test(feat1, feat2)
        feat = self.conv3(feat)
        feat = self.conv4(feat)

        out = self.flow_reg(feat)

        return out.permute(0, 2, 1)


    def _upsample(self, x, size):
        return F.interpolate(x, size=size, mode='nearest')

    def _fusion_test(self, feat1, feat2):
        a = torch.cat((feat1, feat2), dim=1).shape
        return torch.cat((feat1, feat2), dim=1)

    def _fusion(self, feat1, feat2, kernel_size=3, max_displacement=5):
        batch_size, n_channel, n_pts = feat1.shape
        half_kernel = kernel_size // 2
        feat_fused = torch.zeros((batch_size, 2 * max_displacement + 1, n_pts)).cuda()

        # indices of patches of scan1
        patch_ids = torch.arange(n_pts).unsqueeze(dim=-1).long()
        window_ids = torch.arange(-half_kernel, half_kernel + 1).long()
        patch_ids = patch_ids + window_ids.unsqueeze(dim=0)
        patch_ids = patch_ids.clamp(min=0, max=n_pts - 1)

        patch1 = feat1[:, :, patch_ids.reshape(-1)].reshape(batch_size, n_channel, n_pts, kernel_size)
        patch1 = patch1.permute(0, 1, 3, 2) #Permute the shape into (batch_size, n_channel, kernel_size, n_pts)
        patch1 = patch1.reshape(batch_size, n_channel * kernel_size, n_pts)

        patch2 = feat2[:, :, patch_ids.reshape(-1)].reshape(batch_size, n_channel, n_pts, kernel_size)
        patch2 = patch2.permute(0, 1, 3, 2)
        patch2 = patch2.reshape(batch_size, n_channel * kernel_size, n_pts)

        patch_corr = torch.matmul(patch1.permute(0, 2, 1), patch2) # correlation matrix for each pair of patches

        # indices of points within max displacement
        patch2_ids = torch.arange(n_pts).unsqueeze(dim=-1).long()
        displacement_ids = torch.arange(-max_displacement, max_displacement + 1).long()
        patch2_ids = patch2_ids + displacement_ids.unsqueeze(dim=0)
        patch2_ids = patch2_ids.clamp(min=0, max=n_pts - 1)
        patch1_ids = torch.arange(n_pts).unsqueeze(dim=-1).expand_as(patch2_ids).long()
        patch_corr_ids = torch.stack((patch1_ids, patch2_ids), dim=2).reshape(-1, 2)

        # mask = torch.zeros(n_pts, n_pts).float().cuda()
        # mask[patch_corr_ids[:, 0], patch_corr_ids[:, 1]] = 1.0
        # patch_corr = patch_corr * mask

        corr_max_displacement = patch_corr[:, patch_corr_ids[:, 0], patch_corr_ids[:, 1]].reshape(batch_size, n_pts, -1)
        feat_fused = corr_max_displacement.permute(0, 2, 1)

        a = feat_fused / feat_fused[0, 0]

        return feat_fused