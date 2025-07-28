import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

def mask_chamfer_distance(src,
                     dst,
                     mask,
                     src_weight=1.0,
                     dst_weight=1.0,
                     criterion_mode='l2',
                     reduction='mean',
                     dynamic_weight=False):
    """Calculate Chamfer Distance of two sets.

    Args:
        src (torch.Tensor): Source set with shape [B, N, C] to
            calculate Chamfer Distance.
        dst (torch.Tensor): Destination set with shape [B, M, C] to
            calculate Chamfer Distance.
        src_weight (torch.Tensor or float): Weight of source loss.
        dst_weight (torch.Tensor or float): Weight of destination loss.
        criterion_mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.

    Returns:
        tuple: Source and Destination loss with the corresponding indices.

            - loss_src (torch.Tensor): The min distance \
                from source to destination.
            - loss_dst (torch.Tensor): The min distance \
                from destination to source.
            - indices1 (torch.Tensor): Index the min distance point \
                for each point in source to destination.
            - indices2 (torch.Tensor): Index the min distance point \
                for each point in destination to source.
    """

    if criterion_mode == 'smooth_l1':
        criterion = smooth_l1_loss
    elif criterion_mode == 'l1':
        criterion = l1_loss
    elif criterion_mode == 'l2':
        criterion = mse_loss
    else:
        raise NotImplementedError
    
    assert src.size(0) == dst.size(0)
    assert mask.size()[:2] == dst.size()[:2]

    src_expand = src.unsqueeze(2).repeat(1, 1, dst.shape[1], 1)
    dst_expand = dst.unsqueeze(1).repeat(1, src.shape[1], 1, 1)
    # print(src.size(), dst.size())
    # src_expand = src.unsqueeze(2).expand(-1, -1, dst.shape[1], -1)
    # dst_expand = dst.unsqueeze(1).expand(-1, src.shape[1], -1, -1)

    distance = criterion(src_expand, dst_expand, reduction='none').sum(-1)
    mask_expand = mask.unsqueeze(dim=1).expand_as(distance)

    max_dist = distance.max().detach()
    new_distance = distance - mask_expand*max_dist

    src2dst_distance, indices1 = torch.min(new_distance, dim=2)  # (B,N)

    src2dst_distance = src2dst_distance + max_dist
    dst2src_distance, indices2 = torch.min(distance, dim=1)  # (B,M)

    loss_src = (src2dst_distance * src_weight)
    loss_dst = (dst2src_distance * dst_weight)

    if dynamic_weight:
        dy_weight = mask.sum(dim=1, keepdim=True)
        dy_weight = dy_weight/mask.shape[1]
        # print(dy_weight.max(), dy_weight.min())
        loss_src = loss_src * dy_weight

    if reduction == 'sum':
        loss_src = torch.sum(loss_src)
        loss_dst = torch.sum(loss_dst * mask)
    elif reduction == 'mean':
        loss_src = torch.mean(loss_src)
        loss_dst = (loss_dst*mask).sum()/mask.sum()
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError

    return loss_src, loss_dst, indices1, indices2


# @LOSSES.register_module()
class MaskChamferDistance(nn.Module):
    """Calculate Chamfer Distance of two sets.

    Args:
        mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_src_weight (float): Weight of loss_source.
        loss_dst_weight (float): Weight of loss_target.
    """

    def __init__(self,
                 mode='l1',
                 reduction='mean',
                 loss_src_weight=1.0,
                 loss_dst_weight=1.0,
                 dynamic_weight=False,):
        super(MaskChamferDistance, self).__init__()

        assert mode in ['smooth_l1', 'l1', 'l2']
        assert reduction in ['none', 'sum', 'mean']
        self.mode = mode
        self.reduction = reduction
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight

        self.dynamic_weight = dynamic_weight

    def forward(self,
                source,
                target,
                mask,
                src_weight=1.0,
                dst_weight=1.0,
                reduction_override=None,
                return_indices=False,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            source (torch.Tensor): Source set with shape [B, N, C] to
                calculate Chamfer Distance.
            target (torch.Tensor): Destination set with shape [B, M, C] to
                calculate Chamfer Distance.
            src_weight (torch.Tensor | float, optional):
                Weight of source loss. Defaults to 1.0.
            dst_weight (torch.Tensor | float, optional):
                Weight of destination loss. Defaults to 1.0.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.
            return_indices (bool, optional): Whether to return indices.
                Defaults to False.

        Returns:
            tuple[torch.Tensor]: If ``return_indices=True``, return losses of \
                source and target with their corresponding indices in the \
                order of ``(loss_source, loss_target, indices1, indices2)``. \
                If ``return_indices=False``, return \
                ``(loss_source, loss_target)``.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_source, loss_target, indices1, indices2 = mask_chamfer_distance(
            source, target, mask, src_weight, dst_weight, self.mode, reduction, self.dynamic_weight)

        loss_source *= self.loss_src_weight
        loss_target *= self.loss_dst_weight

        if return_indices:
            return loss_source, loss_target, indices1, indices2
        else:
            return loss_source, loss_target


def chamfer_distance(src,
                     dst,
                     src_weight=1.0,
                     dst_weight=1.0,
                     criterion_mode='l1',
                     reduction='mean'):
    """Calculate Chamfer Distance of two sets.

    Args:
        src (torch.Tensor): Source set with shape [B, N, C] to
            calculate Chamfer Distance.
        dst (torch.Tensor): Destination set with shape [B, M, C] to
            calculate Chamfer Distance.
        src_weight (torch.Tensor or float): Weight of source loss.
        dst_weight (torch.Tensor or float): Weight of destination loss.
        criterion_mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.

    Returns:
        tuple: Source and Destination loss with the corresponding indices.

            - loss_src (torch.Tensor): The min distance \
                from source to destination.
            - loss_dst (torch.Tensor): The min distance \
                from destination to source.
            - indices1 (torch.Tensor): Index the min distance point \
                for each point in source to destination.
            - indices2 (torch.Tensor): Index the min distance point \
                for each point in destination to source.
    """

    if criterion_mode == 'smooth_l1':
        criterion = smooth_l1_loss
    elif criterion_mode == 'l1':
        criterion = l1_loss
    elif criterion_mode == 'l2':
        criterion = mse_loss
    else:
        raise NotImplementedError

    src_expand = src.unsqueeze(2).repeat(1, 1, dst.shape[1], 1)
    dst_expand = dst.unsqueeze(1).repeat(1, src.shape[1], 1, 1)
    # print(src.size(), dst.size())
    # src_expand = src.unsqueeze(2).expand(-1, -1, dst.shape[1], -1)
    # dst_expand = dst.unsqueeze(1).expand(-1, src.shape[1], -1, -1)

    distance = criterion(src_expand, dst_expand, reduction='none').sum(-1)
    src2dst_distance, indices1 = torch.min(distance, dim=2)  # (B,N)
    dst2src_distance, indices2 = torch.min(distance, dim=1)  # (B,M)

    loss_src = (src2dst_distance * src_weight)
    loss_dst = (dst2src_distance * dst_weight)

    if reduction == 'sum':
        loss_src = torch.sum(loss_src)
        loss_dst = torch.sum(loss_dst)
    elif reduction == 'mean':
        loss_src = torch.mean(loss_src)
        loss_dst = torch.mean(loss_dst)
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError

    return loss_src, loss_dst, indices1, indices2

class ChamferLoss(nn.Module):
    def __init__(self, args):
        super(ChamferLoss, self).__init__()
        self.loss_dict = {}
        self.coor_loss = MaskChamferDistance()
        self.num_loss = nn.SmoothL1Loss(reduction='none', beta=1.0)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.chamfer_weight = args['chamfer_weight']
        self.pc_num_weight = args['pc_num_weight']


    def forward(self, output_dict, target_dict=None):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        
        # Point Cloud Recon
        pred_coor = output_dict['pred_coor']
        gt_coor = output_dict['gt_coor'].detach()
        chamfer_mask = output_dict['chamfer_mask'].detach()
        gt_mask = output_dict['gt_mask'].detach()

        # print(pred_coor.size(), gt_coor.size(), chamfer_mask.size(), gt_mask.size())
        # torch.Size([1, 60, 100, 176]) torch.Size([1, 5, 180, 100, 176]) torch.Size([1, 5, 100, 176]) torch.Size([1, 100, 176])
        # torch.Size([1, 60, 100, 176]) torch.Size([1, 5, 3, 100, 176]) torch.Size([1, 5, 100, 176]) torch.Size([1, 100, 176])
        # exit()

        chamfer_loss = self.get_coor_loss(pred_coor, gt_coor, gt_mask, chamfer_mask)


        # Point Cloud Num Recon
        pred_num = output_dict['pred_num']
        gt_num = output_dict['gt_num'].detach()
        
        pc_num_loss = self.get_num_loss(pred_num, gt_num, gt_mask)

        total_loss = self.chamfer_weight * chamfer_loss + self.pc_num_weight * pc_num_loss

        self.loss_dict.update({'total_loss': total_loss,
                               'chamfer_loss': chamfer_loss,
                               'pc_num_loss': pc_num_loss})

        return total_loss
    
    def get_kl_loss(self, pred, target, mask):
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        mask = mask.view(mask.size(0), -1)
        pred = F.log_softmax(pred * mask, dim=1)
        target = F.softmax(target * mask, dim=1)

        loss = self.kl_loss(pred, target)
        return loss

    def get_num_loss(self, pred, target, mask):
        bs = pred.shape[0]
        # print(pred.shape)
        # print(target.shape) # 4, 1, 100, 352
        # exit()
        loss = self.num_loss(pred, target).squeeze()
        if bs == 1:
            loss = loss.unsqueeze(dim=0)

        assert loss.size() == mask.size()
        loss = (loss * mask).sum() / mask.sum()
        
        return loss

    def get_js_div_loss(self, pred, target, mask):
        return js_divergence(pred, target, mask)
    
    def get_coor_loss(self, pred, target, mask, chamfer_mask):
        bs, d, _, h, w = target.shape
        target = target.reshape(bs, -1, h, w)
        target = target.permute(0, 2, 3, 1)
        pred = pred.permute(0, 2, 3, 1)
        chamfer_mask = chamfer_mask.permute(0, 2, 3, 1)

        mask = mask.squeeze().bool()
        if bs == 1:
            mask = mask.unsqueeze(dim=0)
        pred = pred[mask]
        target = target[mask]

        chamfer_mask = chamfer_mask[mask]

        pred = pred.reshape(-1, 3, 20).permute(0, 2, 1)
        target = target.reshape(-1, d, 3)
        loss_source, loss_target = self.coor_loss(pred, target, chamfer_mask)

        loss = loss_source + loss_target
        return loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        chamfer_loss = self.loss_dict['chamfer_loss']
        pc_num_loss = self.loss_dict['pc_num_loss']
        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || Chamfer Loss: %.4f"
                " || PcNum Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), chamfer_loss.item(), pc_num_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || Chamfer Loss: %.4f"
                " || PcNum Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), chamfer_loss.item(), pc_num_loss.item()))


        writer.add_scalar('Chamfer_loss', chamfer_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('PcNum_loss', pc_num_loss.item(),
                          epoch*batch_len + batch_id)