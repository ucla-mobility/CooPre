# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch.nn as nn

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelResBackBone8xMAE
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone


class SecondCooPre(nn.Module):
    def __init__(self, args):
        super(SecondCooPre, self).__init__()
        num_class = args['num_class']
        self.batch_size = args['batch_size']
        # mean_vfe
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        # sparse 3d backbone
        self.backbone_3d = VoxelResBackBone8xMAE(args['backbone_3d'],
                                           4, args['grid_size'])
        # height compression
        self.height_compression = HeightCompression(args['height_compression'])
        # base ben backbone
        self.backbone_2d = BaseBEVBackbone(args['base_bev_backbone'], 256)


    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        bev_voxel_features = data_dict['processed_lidar']['bev_voxel_features']
        bev_voxel_coords = data_dict['processed_lidar']['bev_voxel_coords']
        bev_voxel_num_points = data_dict['processed_lidar']['bev_voxel_num_points']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'voxel_features_original': bev_voxel_features,
                      'voxel_coords_original': bev_voxel_coords,
                      'voxel_num_points_original': bev_voxel_num_points,
                      'batch_size': self.batch_size}

        batch_dict = self.mean_vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)

        # Point Cloud Recon
        pc_dict = batch_dict['re_dict']

        pred_coor = pc_dict['pred_coor']
        gt_coor = pc_dict['gt_coor']
        chamfer_mask = pc_dict['chamfer_mask']

        # Point Cloud Num Recon
        pred_num = pc_dict['pred_num']
        gt_num = pc_dict['gt_num']
        gt_mask = pc_dict['gt_mask']

        output_dict = {'pred_coor': pred_coor,
                       'gt_coor': gt_coor,
                       'gt_mask': gt_mask,
                       'chamfer_mask': chamfer_mask,
                       'pred_num': pred_num,
                       'gt_num': gt_num,
                        }

        return output_dict
