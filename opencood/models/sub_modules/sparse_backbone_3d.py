from functools import partial
import numpy as np
import spconv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try: # spconv1
    from spconv import SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor
except: # spconv2
    from spconv.pytorch import  SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor, SparseModule

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = SparseSequential(
            SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        if 'num_features_out' in self.model_cfg:
            self.num_point_features = self.model_cfg['num_features_out']
        else:
            self.num_point_features = 128
        self.conv_out = SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            SparseConv3d(64, self.num_point_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.num_point_features),
            nn.ReLU(),
        )

        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], \
                                       batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.input_channels = input_channels
        print(f'input channels: {input_channels}')

        self.conv_input = SparseSequential(
            SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        # last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        voxel_features = voxel_features[:, :self.input_channels]
        batch_size = batch_dict['batch_size']
        input_sp_tensor = SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict

class VoxelResBackBone8xMAE(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.mask_ratio = model_cfg['mask_ratio']
        self.grid = model_cfg['grid']
        
        self.sparse_shape = grid_size[::-1] + [1, 0, 0] # [41, 800, 2816]

        self.conv_input = SparseSequential(
            SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        self.conv_out = SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        
        self.num_point_features = 16                

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.coor_conv = nn.Conv2d(256, 3*20, 1)
        self.num_conv = nn.Conv2d(256, 1, 1)

        down_factor = 8 #model_cfg['down_factor']
        self.down_factor = down_factor
        self.unshuffle = torch.nn.PixelUnshuffle(down_factor)
        voxel_size = model_cfg['voxel_size']
        point_cloud_range = model_cfg['lidar_range']
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0] 
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

        self.mask_token = nn.Parameter(torch.zeros(1,4))

        self.forward_re_dict = {}
      
    def decode_feat(self, feats, mask=None):
        # feats = feats[mask]
        if mask is not None:
            bs, c, h, w = feats.shape
            # print(mask.shape)
            mask_tokens = self.mask_token.view(1, -1, 1, 1).expand(bs, -1, h, w)
            w = mask.unsqueeze(dim=1).expand_as(mask_tokens)
            feats = feats + w * mask_tokens

        x = self.decoder(feats)
        bs, c, h, w = x.shape
        # x = feats
        coor = self.coor_conv(x)
        num = self.num_conv(x)
        # x = x.reshape(bs, )
        return coor, num


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, coors, num_points = batch_dict['voxel_features'], batch_dict['voxel_coords'], batch_dict['voxel_num_points']
        # print(voxel_features.size())
        # draw_point(voxel_features.cpu().numpy(), './imgs/ori.jpg')

        coor_down_sample = coors.int().detach().clone()
        coor_down_sample[:, 1:] = coor_down_sample[:, 1:]//(self.down_factor * self.grid)
        coor_down_sample[:, 1] = coor_down_sample[:, 1]//(coor_down_sample[:, 1].max()*2)

        unique_coor_down_sample, inverse_index = torch.unique(coor_down_sample, return_inverse=True, dim=0)

        select_ratio = 1 - self.mask_ratio # ratio for select voxel
        nums = unique_coor_down_sample.shape[0]
        
        len_keep = int(nums * select_ratio)

        noise = torch.rand(nums, device=voxel_features.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise)
        ids_restore = torch.argsort(ids_shuffle)

        keep = ids_shuffle[:len_keep]

        unique_keep_bool = torch.zeros(nums).to(voxel_features.device).detach()
        unique_keep_bool[keep] = 1
        # unique_mask_bool = unique_mask_bool.bool()
        ids_keep = torch.gather(unique_keep_bool, 0, inverse_index)
        ids_keep = ids_keep.bool()

        ids_mask = ~ids_keep

        batch_size = batch_dict['batch_size']
        
        ### mask
        voxel_features_mask, voxel_coords_mask = voxel_features[ids_mask,:], coors[ids_mask,:]

        voxel_fratures_mask_one = torch.ones(voxel_features_mask.shape[0],1).to(voxel_features_mask.device).detach()
        pts_mask = SparseConvTensor(
            voxel_fratures_mask_one,
            voxel_coords_mask.int(),
            self.sparse_shape,
            batch_size
        ).dense()

        pts_mask = pts_mask.detach()

        pts_mask = self.unshuffle(pts_mask)
        bev_mask = pts_mask.squeeze(dim=1).max(dim=1)[0]# grid level masking 
        # bev_mask = bev_mask.max(dim=1)[0]
        self.forward_re_dict['gt_mask'] = bev_mask

        #### gt num
        pts_gt_num = SparseConvTensor(
            num_points.view(-1, 1).detach(),
            coors.int(),
            self.sparse_shape,
            batch_size
        ).dense()
        bs, _, d, h, w = pts_gt_num.shape
        # print('num shape 1', pts_gt_num.shape)
        pts_gt_num = self.unshuffle(pts_gt_num.reshape(bs, d, h, w))
        pts_gt_num = pts_gt_num.sum(dim=1, keepdim=True) / self.down_factor**2
        pts_gt_num = pts_gt_num.detach()
        self.forward_re_dict['gt_num'] = pts_gt_num

        ### input
        voxel_features_partial, voxel_coords_partial = voxel_features[ids_keep,:], coors[ids_keep,:]
        # print(voxel_features_partial.size())
        # draw_point(voxel_features_partial.cpu().numpy(), './imgs/unmasked.jpg')

        average_features = self.mask_token.repeat(voxel_features_mask.size(0), 1)

        voxel_features_partial = torch.cat([voxel_features_partial, average_features], dim=0)
        voxel_coords_partial = torch.cat([voxel_coords_partial, voxel_coords_mask], dim=0)

        input_sp_tensor = SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        # print(voxel_features_partial.size(), voxel_coords_partial.size(), self.sparse_shape)

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        feats = out.dense()
        bs, c, d, h, w = feats.shape
        feats = feats.reshape(bs, -1, h, w)

        pred_coor, pred_num = self.decode_feat(feats)
        
        # print(pred_coor.size())
        # exit()
        self.forward_re_dict['pred_coor'] = pred_coor
        self.forward_re_dict['pred_num'] = pred_num
        # print(pred_coor)
        # print(pred_coor.size())
        # exit()
    
        voxels_large, num_points_large, coors_large = batch_dict['voxel_features_original'], batch_dict['voxel_num_points_original'], batch_dict['voxel_coords_original']

        f_center = torch.zeros_like(voxels_large[:, :, :3])

        f_center[:, :, 0] = (voxels_large[:, :, 0] - (coors_large[:, 3].unsqueeze(dim=1) * self.vx + self.x_offset)) / self.vx
        f_center[:, :, 1] = (voxels_large[:, :, 1] - (coors_large[:, 2].unsqueeze(dim=1) * self.vy + self.y_offset)) / self.vy
        f_center[:, :, 2] = (voxels_large[:, :, 2]) / self.vz

        # exit() 
        voxel_count = f_center.shape[1]


        mask_num = get_paddings_indicator(num_points_large, voxel_count, axis=0)
        mask_num = torch.unsqueeze(mask_num, -1).type_as(f_center)
        f_center *= mask_num

        sparse_shape = [1, self.sparse_shape[1]//self.down_factor, self.sparse_shape[2]//self.down_factor]

        chamfer_mask = SparseConvTensor(
            features=mask_num.squeeze().detach(),
            indices=coors_large.int(),
            spatial_shape=sparse_shape,
            batch_size=batch_size
        ).dense()
        
        self.forward_re_dict['chamfer_mask'] = chamfer_mask.sum(dim=2)
        
        n, m, _ = f_center.shape
        f_center = f_center.reshape(n, -1)

        pts_gt_coor = SparseConvTensor(
            features = f_center.detach(),
            indices=coors_large.int(),
            spatial_shape=sparse_shape,
            batch_size=batch_size
        ).dense() # 

        bs, _, d, h, w = pts_gt_coor.shape
        pts_gt_coor = pts_gt_coor.reshape(bs, m, -1, h, w)
        self.forward_re_dict['gt_coor'] = pts_gt_coor
        batch_dict['re_dict'] = self.forward_re_dict

        return batch_dict

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


def draw_point(points, path, size=20, s=2):
    plt.figure(figsize=(size, size))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    ax.axis('off')
    # points = points[(points[:, 0]>-40) & (points[:, 0]<40) &(points[:, 0]>-40) &(points[:, 1]<40)]
    if points.shape[1]<3:
        # ax.scatter(points[:, 1], points[:, 0], s=0.5, c='b', alpha=0.5)
        ax.scatter(points[:, 1], points[:, 0], s=s, c='b', alpha=0.5)
    else:
        # ax.scatter(points[:, 1], points[:, 0], s=0.5, c=points[:, 2], alpha=0.5)
        ax.scatter(points[:, 1], points[:, 0], s=s, c=points[:, 3], alpha=0.5)
    print(path)
    plt.savefig(path)