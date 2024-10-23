import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_utils as pt_utils
from helper_tool import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix
import math


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.in_dim = in_dim

        self.num_heads = 8
        self.dim = dim
        self.k = 64

        # self.linear_proj = nn.Linear(self.dim, self.in_dim)
        self.q_linear = nn.Linear(self.dim, self.in_dim)

        self.linear_0 = nn.Linear(self.in_dim, self.k, bias=False)

        self.linear_1 = nn.Linear(self.k, self.in_dim)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.in_dim, self.in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # idn = self.linear_proj(x)

        x = self.q_linear(x)
        idn = x[:]

        x = x.view(B, N, -1)

        attn = self.linear_0(x)
        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        attn = self.attn_drop(attn)
        x = self.linear_1(attn)

        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = idn + x  # because the original x has different size with current x, use v to do skip connection
        return x.permute(0, 2, 1).unsqueeze(-1)


class Proposed_network(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if (config.name == 'Car'):
            self.class_weights = DP.get_class_weights('Car')
            self.fc0 = nn.Linear(6, 8)
            self.fc0_bath = nn.BatchNorm1d(8, eps=1e-6, momentum=0.99)
            self.fc0_acti = nn.LeakyReLU()
            nn.init.constant_(self.fc0_bath.weight, 1.0)
            nn.init.constant_(self.fc0_bath.bias, 0)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block_pooling_angle(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        self.decoder_blocks = nn.ModuleList()
        self.mlp_blocks = nn.ModuleList()
        self.all_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j < config.num_layers - 1:
                d_in = d_out + 2 * self.config.d_out[-j - 2]
                d_uatten_out=d_out
                d_out = 2 * self.config.d_out[-j - 2]
            else:
                d_in = 4 * self.config.d_out[-config.num_layers]
                d_out = 2 * self.config.d_out[-config.num_layers]
                d_uatten_out=d_out
            self.mlp_blocks.append(nn.Sequential(nn.Conv2d(d_out, d_uatten_out, kernel_size=1, stride=1, padding=0, bias=True),nn.BatchNorm2d(d_uatten_out)))
            self.all_blocks.append(nn.Sequential(nn.Conv2d(d_uatten_out, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(1), nn.Sigmoid()))
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))
        self.relu = nn.ReLU(inplace=True)
        self.attn = Attention(d_out, in_dim=d_out, num_heads=1, attn_drop=0.1, proj_drop=0.1)
        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1, 1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1, 1), bn=False, activation=None)

    def forward(self, end_points):
        features = end_points['features']
        features = self.fc0(features)
        features = features.transpose(1, 2)
        features = self.fc0_bath(features)
        features = self.fc0_acti(features)
        features = features.unsqueeze(dim=3)
        f_encoder_list = []
        f_features_list = [features]
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i],end_points['neigh_idx'][i])
            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            f_features_list.append(features)
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        features = self.decoder_0(f_encoder_list[-1])
        f_decoder_list = []
        f_interp_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])
            adjust_features=self.mlp_blocks[j](f_encoder_list[-j - 2])
            f_interp_i=self.all_blocks[j](self.relu(adjust_features+f_interp_i))*f_interp_i
            f_interp_list.append(f_interp_i)
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))
            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        features = self.attn(features.squeeze(-1).permute(0, 2, 1))
        features = self.fc1(features)
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.fc3(features)
        f_out = features.squeeze(3)
        end_points['logits'] = f_out
        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):
        feature = feature.squeeze(dim=3)
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        feature = feature.squeeze(dim=3)
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2,interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)
        return interpolated_features


class Dilated_res_block_pooling_angle(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True)
        self.lfa = Building_block_pooling_angle(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)
        f_pc= self.lfa(xyz, f_pc, neigh_idx)
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)

        return F.leaky_relu(f_pc + shortcut, negative_slope=0.2)


class Att_dis_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in+2, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set, relative_feature_set):
        att_activation = self.fc(relative_feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


def compute_acc(end_points):
    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])
    end_points['acc'] = acc
    return acc, end_points


class IoUCalculator:
    def __init__(self, cfg):
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, end_points):
        logits = end_points['valid_logits']
        labels = end_points['valid_labels']
        pred = logits.max(dim=1)[1]
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()
        val_total_correct = 0
        val_total_seen = 0
        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)
        conf_matrix = confusion_matrix(labels_valid, pred_valid, labels=np.arange(0, self.cfg.num_classes, 1))
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[ n]) != 0:
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])  # 求第n个类的IoU
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set):
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


def compute_loss(end_points, cfg, device):
    logits = end_points['logits']
    labels = end_points['labels']
    logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
    labels = labels.reshape(-1)
    ignored_bool = torch.zeros(len(labels), dtype=torch.bool).to(device)
    valid_idx = ignored_bool == 0
    valid_logits = logits[valid_idx, :]
    valid_labels_init = labels[valid_idx]
    reducing_list = torch.arange(0, cfg.num_classes).long().to(device)
    valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
    loss = get_loss(valid_logits, valid_labels, DP.get_class_weights('every_Car'), device)
    end_points['valid_logits'], end_points[
        'valid_labels'] = valid_logits, valid_labels
    end_points['loss'] = loss
    return loss, end_points


def get_loss(logits, labels, pre_cal_weights, device):
    class_weights = torch.from_numpy(pre_cal_weights).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.reshape([-1]),
                                    reduction='none')
    output_loss = criterion(logits, labels)
    output_loss = output_loss.mean()
    return output_loss


