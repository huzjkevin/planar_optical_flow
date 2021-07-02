import numpy as np

import torch
import torch.nn.functional as F

import src.utils.utils as u
from src.utils.rotate_iou import rotate_iou_gpu_eval

_INPUT_WITH_ANGLE = True


def _model_fn(model, batch):
    tb_dict, rtn_dict = {}, {}
    # unpack data
    input, target = batch["input"], batch["target"]

    # Move data to GPU
    input = torch.from_numpy(input).cuda(non_blocking=True).float()
    target = torch.from_numpy(target).cuda(non_blocking=True).float()

    pred = model(input)

    loss = model.loss_fn(pred, target)

    rtn_dict["pred"] = pred

    return loss, tb_dict, rtn_dict


def _model_eval_fn(model, batch):
    loss, tb_dict, rtn_dict = _model_fn(model, batch)

    target = batch["target"]
    pred = rtn_dict["pred"].data.cpu().numpy()
    det_center = batch["det_center"]
    box_center = batch["box_center"]
    input = batch["input"]
    target_neighbor = batch["target_neighbor"]

    if box_center.shape[1] == 3:
        is_3d = True
    elif box_center.shape[1] == 2:
        is_3d = False

    if is_3d:
        # transform cz back to global
        pred[:, 0] += det_center[:, -1]
        target[:, 0] += det_center[:, -1]

        loss_z = np.abs(pred[:, 0] - target[:, 0])
        loss_dims = np.sum(np.abs(pred[:, 1:-1] - target[:, 1:-1]), axis=1)
        rot_z = batch["rot_z"]
        pred[:, -1] += input[:, 0, -1]  # orientation = input angle + regressed residual
        pred = np.hstack((det_center[:, :2], pred))
        target[:, -1] = rot_z
        target = np.hstack((box_center[:, :2], target))
    else:
        loss_dims = np.sum(np.abs(pred[:, :-1] - target[:, :-1]), axis=1)
        rot_z = batch["rot_z"]
        pred[:, -1] += input[:, 0, -1]  # orientation = input angle + regressed residual
        pred = np.hstack((det_center, pred))
        target[:, -1] = rot_z
        target = np.hstack((box_center[:, :2], target))

    # if _INPUT_WITH_ANGLE:
    #     rot_z = batch["rot_z"]
    #     pred[:, -1] += input[:, 0, -1]  # orientation = input angle + regressed residual
    #     pred = np.hstack((det_center, pred))
    #     target[:, -1] = rot_z
    #     target = np.hstack((box_center, target))
    # else:
    #     pred = np.hstack((det_center, pred, target[:, -1].reshape(-1, 1)))
    #     target = np.hstack((box_center, target))

    # target[:, :2] += det_center
    ious = []
    for i in range(len(pred)):
        iou = rotate_iou_gpu_eval(
            pred[i].reshape(1, -1), target_neighbor[i], is_3d=is_3d
        )
        max_iou = np.max(iou)
        ious.append(max_iou)
    # iou = rotate_iou_gpu_eval(pred, target, is_3d=is_3d)
    loss_ori = np.abs(pred[:, -1] - target[:, -1])  # rot_z as last channel

    rtn_dict = {
        # "iou": np.mean(np.diag(iou)),
        "iou": np.mean(ious),
        "loss_z": np.mean(loss_z),
        "loss_dim": np.mean(loss_dims),
        "loss_ori": np.mean(loss_ori),
    }

    return loss, tb_dict, rtn_dict
