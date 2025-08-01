# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Transform points to voxels using sparse conv library
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from cumm import tensorview as tv
from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor


class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(SpVoxelPreprocessor, self).__init__(preprocess_params,
                                                  train)
        self.spconv = 1
        try:
            # spconv v1.x
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
        except:
            # spconv v2.x
            from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
            self.spconv = 2
        self.lidar_range = self.params['cav_lidar_range']
        self.voxel_size = self.params['args']['voxel_size']
        self.max_points_per_voxel = self.params['args']['max_points_per_voxel']

        if train:
            self.max_voxels = self.params['args']['max_voxel_train']
        else:
            self.max_voxels = self.params['args']['max_voxel_test']

        grid_size = (np.array(self.lidar_range[3:6]) -
                     np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        # use sparse conv library to generate voxel
        if self.spconv == 1:
            self.voxel_generator = VoxelGenerator(
                voxel_size=self.voxel_size,
                point_cloud_range=self.lidar_range,
                max_num_points=self.max_points_per_voxel,
                max_voxels=self.max_voxels
            )
        else:
            self.voxel_generator = VoxelGenerator(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.lidar_range,
                max_num_points_per_voxel=self.max_points_per_voxel,
                num_point_features=4,
                max_num_voxels=self.max_voxels
            )

        self.bev_voxel_generator = None
        if 'bev_args' in self.params:
            self.bev_voxel_size = self.params['bev_args']['voxel_size']
            self.bev_max_points_per_voxel = self.params['bev_args']['max_points_per_voxel']
            self.bev_voxel_generator = VoxelGenerator(
                vsize_xyz=self.bev_voxel_size,
                coors_range_xyz=self.lidar_range,
                max_num_points_per_voxel=self.bev_max_points_per_voxel,
                num_point_features=4,
                max_num_voxels=self.max_voxels
            )

    def preprocess(self, pcd_np):
        # self.draw_point(pcd_np, 'test_original_pcd.png')
        data_dict = {}
        if self.spconv == 1:
            voxel_output = self.voxel_generator.generate(pcd_np)
        else:
            pcd_tv = tv.from_numpy(pcd_np)
            voxel_output = self.voxel_generator.point_to_voxel(pcd_tv)
            if self.bev_voxel_generator is not None:
                bev_voxel_output = self.bev_voxel_generator.point_to_voxel(pcd_tv)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], \
                voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if self.bev_voxel_generator is not None:
            if isinstance(voxel_output, dict):
                bev_voxels, bev_coordinates, bev_num_points = \
                    bev_voxel_output['voxels'], bev_voxel_output['coordinates'], \
                    bev_voxel_output['num_points_per_voxel']
            else:
                bev_voxels, bev_coordinates, bev_num_points = bev_voxel_output

        if self.spconv == 2:
            voxels = voxels.numpy()
            coordinates = coordinates.numpy()
            num_points = num_points.numpy()
            if self.bev_voxel_generator is not None:
                bev_voxels = bev_voxels.numpy()
                bev_coordinates = bev_coordinates.numpy()
                bev_num_points = bev_num_points.numpy()

        data_dict['voxel_features'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points

        # self.draw_point(voxels, 'test_original_voxel.png')
        # exit()

        if self.bev_voxel_generator is not None:
            data_dict['bev_voxel_features'] = bev_voxels
            data_dict['bev_voxel_coords'] = bev_coordinates
            data_dict['bev_voxel_num_points'] = bev_num_points

        return data_dict

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """

        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represent a single frame.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = []
        voxel_num_points = []
        voxel_coords = []
        bev_voxel_features = []
        bev_voxel_num_points = []
        bev_voxel_coords = []

        for i in range(len(batch)):
            voxel_features.append(batch[i]['voxel_features'])
            voxel_num_points.append(batch[i]['voxel_num_points'])
            coords = batch[i]['voxel_coords']
            voxel_coords.append(
                np.pad(coords, ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))
            if 'bev_voxel_features' in batch[i]:
                bev_voxel_features.append(batch[i]['bev_voxel_features'])
                bev_voxel_num_points.append(batch[i]['bev_voxel_num_points'])
                coords = batch[i]['bev_voxel_coords']
                bev_voxel_coords.append(
                    np.pad(coords, ((0, 0), (1, 0)),
                        mode='constant', constant_values=i))


        voxel_num_points = torch.from_numpy(np.concatenate(voxel_num_points))
        voxel_features = torch.from_numpy(np.concatenate(voxel_features))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))
        if len(bev_voxel_num_points) > 0:
            bev_voxel_num_points = torch.from_numpy(np.concatenate(bev_voxel_num_points))
            bev_voxel_features = torch.from_numpy(np.concatenate(bev_voxel_features))
            bev_voxel_coords = torch.from_numpy(np.concatenate(bev_voxel_coords))
        
        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points,
                'bev_voxel_features': bev_voxel_features,
                'bev_voxel_coords': bev_voxel_coords,
                'bev_voxel_num_points': bev_voxel_num_points,
                }


    @staticmethod
    def collate_batch_dict(batch: dict):
        """
        Collate batch if the batch is a dictionary,
        eg: {'voxel_features': [feature1, feature2...., feature n]}

        Parameters
        ----------
        batch : dict

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = \
            torch.from_numpy(np.concatenate(batch['voxel_features']))
        voxel_num_points = \
            torch.from_numpy(np.concatenate(batch['voxel_num_points']))
        coords = batch['voxel_coords']
        voxel_coords = []

        for i in range(len(coords)):
            voxel_coords.append(
                np.pad(coords[i], ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        if 'bev_voxel_features' in batch:
            bev_voxel_features = torch.from_numpy(np.concatenate(batch['bev_voxel_features']))
            bev_voxel_num_points = torch.from_numpy(np.concatenate(batch['bev_voxel_num_points']))
            coords = batch['bev_voxel_coords']
            bev_voxel_coords = []
            for i in range(len(coords)):
                bev_voxel_coords.append(
                    np.pad(coords[i], ((0, 0), (1, 0)),
                        mode='constant', constant_values=i))
            bev_voxel_coords = torch.from_numpy(np.concatenate(bev_voxel_coords))
        else:
            bev_voxel_features, bev_voxel_coords, bev_voxel_num_points = [], [], []

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points,
                'bev_voxel_features': bev_voxel_features,
                'bev_voxel_coords': bev_voxel_coords,
                'bev_voxel_num_points': bev_voxel_num_points,
                }
    
    def draw_point(self, points, path, size=20, s=2):
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
        if len(points.shape) > 2:
            points = torch.tensor(points)
            points_mean = points[:, :, :].sum(dim=1, keepdim=False)
            ax.scatter(points_mean[:, 1], points_mean[:, 0], s=s, c='b', alpha=0.5)

        elif points.shape[1]<3:
            # ax.scatter(points[:, 1], points[:, 0], s=0.5, c='b', alpha=0.5)
            ax.scatter(points[:, 1], points[:, 0], s=s, c='b', alpha=0.5)
        else:
            # ax.scatter(points[:, 1], points[:, 0], s=0.5, c=points[:, 2], alpha=0.5)
            ax.scatter(points[:, 1], points[:, 0], s=s, c=points[:, 2], alpha=0.5)
        print(path)
        plt.savefig(path)
