from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor
import numpy as np
from rein.models.utils.prototype_dist_estimator import prototype_dist_estimator
from rein.models.utils.differ_estimator import differ_estimator
from rein.models.utils.mynn import initialize_weights
import kmeans1d
from mmseg.utils import SampleList
from typing import List, Tuple


@MODELS.register_module()
class SAMRelations(nn.Module):
    def __init__(
        self
    ) -> None:
        super().__init__()
        self.create_model()

    def create_model(self):
        self.Proto = []
        self.Differ = []
        self.inter_relation_matrix = nn.Parameter(torch.empty([32, 19, 19]))
        self.inter_relation_matrix.requires_grad = False

        self.intra_relation_matrix_layer = nn.Parameter(torch.zeros([32, 19, 1024]))
        self.intra_relation_matrix_layer.requires_grad = False

        for i in range(32):
            self.Proto.append(prototype_dist_estimator(feature_num=1280))
            self.Differ.append(differ_estimator(feature_num=19))

        # SAM
        self.mapping_layers = nn.Sequential(
                          nn.Conv2d(19, 512, kernel_size=1, bias=False),
                          nn.Conv2d(512, 1280, kernel_size=1, bias=False),
                          nn.ReLU(inplace=True)
                          )
        self.mapping_layers2 = nn.Sequential(
                nn.Conv2d(1280, 512, kernel_size=1, bias=False),
                nn.Conv2d(512, 19, kernel_size=1, bias=False)
            )
        # SAM


    def calc_proto(self, feats, gt, idx):
        Bh, Ch, Hh, Wh = feats.size()
        src_mask = F.interpolate(gt.unsqueeze(1).float(), size=(Hh, Wh), mode='nearest').squeeze(
            0).long()
        src_mask = src_mask.contiguous().view(Bh * Hh * Wh, )
        assert not src_mask.requires_grad

        feats = feats.permute(0, 2, 3, 1).contiguous().view(Bh * Hh * Wh, Ch)
        self.Proto[idx].update(features=feats.detach(), labels=src_mask)

    def inter_class_relation_calib_matrix(self, feat, proto, gt, idx):
        # ----------------- get_class_feat -------------------------------
        feats = feat
        cls_feat = torch.zeros(19, 1280).cuda()  # add SAM
        Bh, Ch, Hh, Wh = feats.size()
        src_mask = F.interpolate(gt.unsqueeze(1).float(), size=(Hh, Wh), mode='nearest').squeeze(
            0).long()  # add
        src_mask = src_mask.contiguous().view(Bh * Hh * Wh, )
        assert not src_mask.requires_grad
        feats = feats.permute(0, 2, 3, 1).contiguous().view(Bh * Hh * Wh, Ch)
        mask = (src_mask != 255)
        labels = src_mask[mask]
        features = feats[mask]
        ids_unique = labels.unique()
        for i in ids_unique:
            i = i.item()
            mask_i = (labels == i)
            feature = features[mask_i]
            feature = torch.mean(feature, dim=0)
            cls_feat[i, :] = feature
        # ----------------- get_class_feat -------------------------------

        eps = 1e-5
        C, N = cls_feat.shape
        eye = torch.eye(C).cuda()
        # ----------------- get_feat_covariance_matrix -------------------------------
        f_cor = torch.mm(cls_feat, cls_feat.transpose(1, 0)).div(N - 1) + (eps * eye)
        # ----------------- get_feat_covariance_matrix -------------------------------

        # ----------------- get_proto_covariance_matrix -------------------------------
        proto_cor = torch.mm(proto, proto.transpose(1, 0)).div(N - 1) + (eps * eye)
        # ----------------- get_proto_covariance_matrix -------------------------------
        # ----------------- get_diff_matrix -------------------------------
        diff = torch.abs(f_cor - proto_cor)
        with torch.no_grad():
            diff_sp = F.softmax(diff, dim=0)
        self.Differ[idx].update(diff_sp)
        # ----------------- get_diff_matrix -------------------------------
        diff_flatten = torch.flatten(self.Differ[idx].differ)
        clusters, centroids = kmeans1d.cluster(diff_flatten, 5)
        num_sensitive = diff_flatten.size()[0] - clusters.count(0)
        _, indices = torch.topk(diff_flatten, k=int(num_sensitive))
        mask_matrix = torch.flatten(torch.ones(19, 19).cuda()) 
        mask_matrix[indices] = 0
        inter_calib_matrix = mask_matrix.view(19, 19)
        self.inter_relation_matrix[idx].data = f_cor * inter_calib_matrix

    def intra_class_relation_calib_matrix(self, feat, idx):
        B, C, H, W = feat.shape
        feat = feat.contiguous().view(B, -1, C)
        temp = torch.zeros([19, H*W]).cuda()
        for i in range(C):
            fi = feat[:,:,i].unsqueeze(2)
            fi_cor = torch.bmm(fi, fi.transpose(1, 2))
            with torch.no_grad():
                fi_cor_sp = F.softmax(fi_cor, dim=1)
                fi_cor_sp_mean = torch.mean(torch.mean(fi_cor_sp, dim=1), dim=0).unsqueeze(0)
            temp[i] = fi_cor_sp_mean
        self.intra_relation_matrix_layer[idx].data = 0.99 * self.intra_relation_matrix_layer[idx].data + 0.01 * temp


    def forward(
        self, x: Tuple[List[Tensor], List[Tensor]], batch_data_samples: SampleList, i, B, H, W, batch_first=False, has_cls_token=True
    ) -> Tuple[List[Tensor], List[Tensor]]:

        if has_cls_token:
            cls_token, x = torch.tensor_split(x, [1], dim=1)

        x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[0] > 1:
            batch_img_label = np.asarray([data_sample.gt_sem_seg.data.squeeze().cpu().numpy() for data_sample in batch_data_samples])
            gt = torch.from_numpy(batch_img_label).cuda()

        new_x = None
        _, C, _, _ = x.shape
        if x.shape[0] > 1:  # training
            self.calc_proto(x, gt, i)
            self.inter_class_relation_calib_matrix(x, self.Proto[i].Proto, gt, i)
            x_i = x.contiguous().view(B, -1, C)
            x_c = torch.bmm(x_i, self.Proto[i].Proto.unsqueeze(0).repeat(B, 1, 1).transpose(1, 2)).contiguous().view(B, 19, H, W)
            self.intra_class_relation_calib_matrix(x_c, i)
        x_m = self.mapping_layers2(x)
        intra_matrix = None
        intra_matrix = self.intra_relation_matrix_layer[i].data

        x_mB, x_mC, x_mH, x_mW = x_m.shape  # i-th feature size (B X C X H X W)
        x_i_v = x_m.contiguous().view(x_mB, -1, x_mC)  # B X C X H X W > B X C X (H X W)

        intra_atten_weights = torch.bmm(x_i_v, intra_matrix.unsqueeze(0).repeat(x_mB, 1, 1))
        with torch.no_grad():
            intra_atten_weights_sf = F.softmax(intra_atten_weights, dim=-1)
        intra_cab_x_c_i = torch.bmm(intra_atten_weights_sf,
                                    intra_matrix.unsqueeze(0).repeat(x_mB, 1, 1).transpose(1, 2)).contiguous().view(
            x_mB, 19, x_mH, x_mW)

        intra_cab_B, intra_cab_C, intra_cab_H, intra_cab_W = intra_cab_x_c_i.shape  # i-th feature size (B X C X H X W)
        x_i_v = intra_cab_x_c_i.contiguous().view(intra_cab_B, -1, intra_cab_C)  # B X C X H X W > B X C X (H X W)
        relation_matrix = self.inter_relation_matrix[i].data.clone().detach().unsqueeze(0).repeat(intra_cab_B, 1, 1)
        cab_x_c_i = torch.bmm(x_i_v, relation_matrix).contiguous().view(intra_cab_B, intra_cab_C, intra_cab_H, intra_cab_W) # .detach()
        cab_x_i = self.mapping_layers(cab_x_c_i)
        att_x = cab_x_i + x

        new_x = att_x.permute(0, 2, 3, 1).contiguous()
        if has_cls_token:
            new_x = torch.cat([cls_token, new_x], dim=1)
        return new_x