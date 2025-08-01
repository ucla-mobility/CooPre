# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.fuse_modules.v2xvit_basic import V2XTransformer
from opencood.models.fuse_modules.fuse_utils import regroup
class PointPillarIntermediate(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediate, self).__init__()
        num_class = args['num_class']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False
        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = AttFusion(args['base_bev_backbone']['num_filters'][-1])
        # self.fusion_net = SpatialFusion

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'] * num_class * num_class,
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_num'] * num_class,
                                  kernel_size=1)



    def extract_features(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
 
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
            # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(
                spatial_features_2d, "encoder")

        return spatial_features_2d
    
    def fuse_features(self, data):
        spatial_features_2d = data['bev']
        record_len = data['record_len']
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d, "decoder")

        fused_feature = self.fusion_net(spatial_features_2d, record_len)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        spatial_correction_matrix = data_dict['spatial_correction_matrix']
        prior_encoding = \
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        4)

        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4])
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

        # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # # downsample feature to reduce memory
        # if self.shrink_flag:
        #     spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # # compressor
        # if self.compression:
        #     spatial_features_2d = self.naive_compressor(spatial_features_2d)

        fused_feature = self.fusion_net(spatial_features_2d, record_len)
        # fused_feature = self.fusion_net(regroup_feature, mask,
        #                                 spatial_correction_matrix)
        psm = self.cls_head(fused_feature)

        rm = self.reg_head(fused_feature)
        # print(fused_feature.size())
        # print(psm.size())
        # print(rm.size())
        # exit()
        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict