import torch
from src.utils.viz_utils import *
import src.utils.utils as u
import numpy as np
import os
import torch.nn.functional as F
from src.utils.jrdb_transforms import Box3d
from .rotate_iou import rotate_iou_gpu_eval

def model_fn(model, batch):
    reg = 1e-3
    #unpack data
    scan_pair, target_flow = batch["scan_pair"], batch["flow_target_flow"]
    # scan_pair, target_flow, mask = batch["scan_pair"], batch["flow_target_flow"], batch['exclude_mask']
    scan1 = scan_pair[:, 0]
    scan2 = scan_pair[:, 1]

    # input, target_flow, _ = data # code for normalization new normalization
    #Move data to GPU
    scan1 = torch.from_numpy(scan1).cuda(non_blocking=True).float()
    scan2 = torch.from_numpy(scan2).cuda(non_blocking=True).float()
    target_flow = torch.from_numpy(target_flow).cuda(non_blocking=True).float()
    # mask = torch.from_numpy(mask).cuda(non_blocking=True).float()

    pred_flow = model(scan1, scan2)

    loss, epe_batch, aae_batch = model.loss_fn(pred_flow, target_flow)

    return loss

def model_fn_obj_det(model, batch, rtn_result=False):
    tb_dict, rtn_dict = {}, {}

    net_input = batch['input']
    net_input = torch.from_numpy(net_input).cuda(non_blocking=True).float()

    # Forward pass
    model_rtn = model(net_input)
    spatial_drow = len(model_rtn) == 3
    if spatial_drow:
        pred_cls, pred_reg, _ = model_rtn
    else:
        pred_cls, pred_reg = model_rtn

    target_flow_cls, target_flow_reg = batch['target_flow_cls'], batch['target_flow_reg']
    target_flow_cls = torch.from_numpy(target_flow_cls).cuda(non_blocking=True).long()
    target_flow_reg = torch.from_numpy(target_flow_reg).cuda(non_blocking=True).float()

    n_batch, n_pts = target_flow_cls.shape[:2]

    # cls loss
    target_flow_cls = target_flow_cls.view(n_batch * n_pts)
    pred_cls = pred_cls.view(n_batch * n_pts, -1)
    if pred_cls.shape[1] == 1:
        cls_loss = model.cls_loss(torch.sigmoid(pred_cls.squeeze(-1)),
                                  target_flow_cls.float(),
                                  reduction='mean')
    else:
        cls_loss = model.cls_loss(pred_cls, target_flow_cls, reduction='mean')
    total_loss = cls_loss
    tb_dict['cls_loss'] = cls_loss.item()

    # number fg points
    fg_mask = target_flow_cls.ne(0)
    fg_ratio = torch.sum(fg_mask).item() / (n_batch * n_pts)
    tb_dict['fg_ratio'] = fg_ratio

    # reg loss
    if fg_ratio > 0.0:
        target_flow_reg = target_flow_reg.view(n_batch * n_pts, -1)
        pred_reg = pred_reg.view(n_batch * n_pts, -1)
        reg_loss = F.mse_loss(pred_reg[fg_mask], target_flow_reg[fg_mask],
                              reduction='none')
        reg_loss = torch.sqrt(torch.sum(reg_loss, dim=1)).mean()
        total_loss = total_loss + reg_loss
        tb_dict['reg_loss'] = reg_loss.item()

    # # regularization loss for spatial attention
    # if spatial_drow:
    #     att_loss = (-torch.log(pred_flow_sim + 1e-5) * pred_flow_sim).sum(dim=2).mean()  # shannon entropy
    #     tb_dict['att_loss'] = att_loss.item()
    #     total_loss = total_loss + att_loss

    if rtn_result:
        rtn_dict["pred_reg"] = pred_reg.view(n_batch, n_pts, -1)
        rtn_dict["pred_cls"] = pred_cls.view(n_batch, n_pts, -1)

    return total_loss, tb_dict, rtn_dict

def model_fn_dr_spaam(model, batch):
    #unpack data
    cur_scan = batch['scans'][:, -1]
    input = batch['input']
    target_flow = batch['target_flow']
    mask = batch['exclude_mask']

    #Move data to GPU
    input = torch.from_numpy(input).cuda(non_blocking=True).float()
    target_flow = torch.from_numpy(target_flow).cuda(non_blocking=True).float()
    mask = torch.from_numpy(mask).cuda(non_blocking=True).float()
    cur_scan = torch.from_numpy(cur_scan).cuda(non_blocking=True).float()

    pred_cls, pred_reg, pred_flow = model(input, cur_scan)
    loss = model.loss_fn(pred_flow, target_flow, mask=mask)

    #========Test=========
    avg_pred_norm = torch.mean(torch.norm(pred_flow, dim=-1)[mask == 1.0])
    avg_target_norm = torch.mean(torch.norm(target_flow, dim=-1)[mask == 1.0])
    #========Test=========
    return loss, avg_pred_norm, avg_target_norm

def model_fn_Bb_regression(model, batch):
    #unpack data
    input, target = batch['input'], batch['target']
    # det_center = batch['det_center']
    # input_center = batch['input_center']

    #Move data to GPU
    input = torch.from_numpy(input).cuda(non_blocking=True).float()
    target = torch.from_numpy(target).cuda(non_blocking=True).float()

    pred = model(input)
    # pred[pred[..., -1] > np.pi, -1] -= 2 * np.pi
    # pred[pred[..., -1] < -np.pi, -1] += 2 * np.pi
    loss = model.loss_fn(pred, target[:, 2:])

    return loss

def loss_fn_eval(pred_flow, target_flow):
    epe_batch = torch.mean(torch.norm(pred_flow - target_flow, dim=-1), dim=1)
    aae_batch = torch.mean(torch.abs(torch.atan2(pred_flow[..., 0], pred_flow[..., 1]) - torch.atan2(target_flow[..., 0], target_flow[..., 1])), dim=1) * 180 / np.pi
    # loss = torch.mean(epe_batch)

    return epe_batch, aae_batch

def model_fn_eval(model, eval_loader):
    epe_loss = 0.0
    aae_loss = 0.0
    with torch.no_grad():
        for cur_it, batch in enumerate(eval_loader):
            cur_scan = batch['scans'][:, -1]
            input = batch['input']
            target_flow = batch['target_flow']

            input = torch.from_numpy(input).cuda(non_blocking=True).float()
            target_flow = torch.from_numpy(target_flow).cuda(non_blocking=True).float()
            cur_scan = torch.from_numpy(cur_scan).cuda(non_blocking=True).float()

            pred_cls, pred_reg, pred_flow = model(input, cur_scan)

            epe_batch, aae_batch = loss_fn_eval(pred_flow, target_flow)
            epe_loss += torch.mean(epe_batch).item()
            aae_loss += torch.mean(aae_batch).item()

    return epe_loss / len(eval_loader), aae_loss / len(eval_loader)

def eval(model, test_loader, cfg, output_dir=None, tb_logger=None):
    reg = 1e-6
    eval_loss = 0.0
    model.eval()

    if output_dir is not None:
        scans = None
        pred_flows = None
        target_flows = None
        epe_list = None

    with torch.no_grad():
        for cur_it, batch in enumerate(test_loader):
            scan_pair, target_flow = batch["scan_pair"], batch["flow_target_flow"]
            # scan_pair, target_flow, mask = batch["scan_pair"], batch["flow_target_flow"], batch['exclude_mask']

            scan1 = scan_pair[:, 0]
            scan2 = scan_pair[:, 1]
            # input, target_flow, _ = data # code for normalization new normalization
            # Move data to GPU
            scan1 = torch.from_numpy(scan1).cuda(non_blocking=True).float()
            scan2 = torch.from_numpy(scan2).cuda(non_blocking=True).float()
            target_flow = torch.from_numpy(target_flow).cuda(non_blocking=True).float()
            # mask = torch.from_numpy(mask).cuda(non_blocking=True).float()

            # Flow pred_flowiction in canonical frame
            pred_flow = model(scan1, scan2)
            pred_flow[torch.norm(pred_flow, dim=-1) <= reg] = 0.0

            epe_batch, _ = loss_fn_eval(pred_flow, target_flow)

            # Change to global
            # pred_flow = u.canonical_to_global_flow(pred_flow, scan_phi)

            if output_dir is not None:

                if scans is None:
                    scans = scan1.cpu().detach().numpy()
                    pred_flows = pred_flow.cpu().detach().numpy()
                    target_flows = target_flow.cpu().detach().numpy()
                    epe_list = epe_batch.cpu().detach().numpy()
                else:
                    scans = np.vstack([scans, scan1.cpu().detach().numpy()])
                    pred_flows = np.vstack([pred_flows, pred_flow.cpu().detach().numpy()])
                    target_flows = np.vstack([target_flows, target_flow.cpu().detach().numpy()])
                    epe_list = np.hstack([epe_list, epe_batch.cpu().detach().numpy()])

            eval_loss += torch.mean(epe_batch).item()

        print("Eval loss: ", eval_loss / len(test_loader))

    if output_dir is not None:
        print("Plotting flow sequence")
        pred_flow_path = os.path.join(output_dir, "tmp_videos")
        target_flow_path = os.path.join(output_dir, "tmp_videos")
        os.makedirs(pred_flow_path, exist_ok=True)
        os.makedirs(target_flow_path, exist_ok=True)

        plot_sequence_gt_pred_arrow(scans[..., 0], scans[..., 1],
                              path=os.path.join(pred_flow_path, "flow_pred_flow_target_flow.mp4"),
                              pred=pred_flows,
                              target=target_flows,
                              epe_list=epe_list)

def eval_dr_spaam(model, test_loader, cfg, output_dir=None, tb_logger=None):
    reg = 1e-6
    eval_loss = 0.0
    model.eval()
    scan_phi = u.get_laser_phi()

    if output_dir is not None:
        scans = None
        pred_flows = None
        target_flows = None
        odoms = None
        epe_list = None
        aae_list = None

    with torch.no_grad():
        for cur_it, batch in enumerate(test_loader):
            cur_scan = batch['scans'][:, -1]
            input = batch['input']
            target_flow = batch['target_flow']
            odom = batch['odom1']
            # mask = batch['exclude_mask']

            input = torch.from_numpy(input).cuda(non_blocking=True).float()
            target_flow = torch.from_numpy(target_flow).cuda(non_blocking=True).float()

            pred_cls, pred_reg, pred_flow = model(input, torch.from_numpy(cur_scan).cuda(non_blocking=True).float())

            epe_batch, aae_batch = loss_fn_eval(pred_flow, target_flow)

            for i in range(test_loader.batch_size):
                pred_flow[i] = u.canonical_to_global_flow_torch(pred_flow[i], scan_phi)
                target_flow[i] = u.canonical_to_global_flow_torch(target_flow[i], scan_phi)

            if output_dir is not None:

                if scans is None:
                    scans = cur_scan
                    odoms = odom
                    pred_flows = pred_flow.data.cpu().numpy()
                    target_flows = target_flow.data.cpu().numpy()
                    epe_list = epe_batch.data.cpu().numpy()
                    aae_list = aae_batch.data.cpu().numpy()

                else:
                    scans = np.vstack([scans, cur_scan])
                    odoms = np.hstack([odoms, odom])
                    pred_flows = np.vstack([pred_flows, pred_flow.data.cpu().numpy()])
                    target_flows = np.vstack([target_flows, target_flow.data.cpu().numpy()])
                    epe_list = np.hstack([epe_list, epe_batch.data.cpu().numpy()])
                    aae_list = np.hstack([aae_list, aae_batch.data.cpu().numpy()])

            eval_loss += torch.mean(epe_batch).item()

        print("Eval loss: ", eval_loss / len(test_loader))

    if output_dir is not None:
        print("Plotting flow sequence")
        path = os.path.join(output_dir, "tmp_videos")
        os.makedirs(path, exist_ok=True)

        scan_phi = u.get_laser_phi()
        scans_xy = u.rphi_to_xy(scans, np.repeat(scan_phi.reshape(1, -1), len(scans), axis=0))
        scans_xy = np.stack(scans_xy, axis=-1)

        hsv_pred_flows, hsv_target_flows = [], []
        for i in range(len(pred_flows)):
            hsv_pred_flows.append(u.flow_to_hsv(pred_flows[i]))
            hsv_target_flows.append(u.flow_to_hsv(target_flows[i]))
        hsv_pred_flows = np.asarray(hsv_pred_flows)
        hsv_target_flows = np.asarray(hsv_target_flows)

        arrow_path = os.path.join(path, "flow_pred_target_arrow.mp4")
        print('Writing arrow sequence to {}'.format(arrow_path))
        plot_sequence_gt_pred_arrow(scans_xy[..., 0], scans_xy[..., 1],
                                    path=arrow_path,
                                    pred=pred_flows,
                                    target=target_flows,
                                    epe_list=epe_list,
                                    aae_list=aae_list)

        hsv_path = os.path.join(path, "flow_pred_target_hsv.mp4")
        print('Writing hsv sequence to {}'.format(hsv_path))
        plot_sequence_gt_pred_hsv(scans_xy[..., 0], scans_xy[..., 1],
                                  path=hsv_path,
                                  pred=hsv_pred_flows,
                                  target=hsv_target_flows,
                                  epe_list=epe_list,
                                  aae_list=aae_list)

        # plot_flow_fixed_pose(scans,
        #                      np.repeat(scan_phi.reshape(1, -1), len(scans), axis=0),
        #                      odoms.reshape(-1, 1),
        #                      path=hsv_path,
        #                      pred=hsv_pred_flows,
        #                      target=hsv_target_flows,
        #                      epe_list=epe_list,
        #                      aae_list=aae_list)


def eval_person_flow(model, test_loader, cfg, output_dir=None, tb_logger=None):
    reg = 1e-6
    eval_loss = 0.0
    model.eval()
    scan_phi = u.get_laser_phi()
    dets_xy = []
    dets_cls = []
    instance_masks = []

    if output_dir is not None:
        scans = None
        pred_flows = None
        target_flows = None
        odoms = None
        epe_list = None
        aae_list = None

    with torch.no_grad():
        for cur_it, batch in enumerate(test_loader):
            scan = batch['scans'][:, -2]
            input = batch['input']
            target_flow = batch['target_flow']
            odom = batch['odom']
            # mask = batch['exclude_mask']

            input = torch.from_numpy(input).cuda(non_blocking=True).float()
            target_flow = torch.from_numpy(target_flow).cuda(non_blocking=True).float()

            pred_cls, pred_reg, pred_flow = model(input)

            pred_cls = torch.sigmoid(pred_cls[0]).data.cpu().numpy()
            pred_reg = pred_reg[0].data.cpu().numpy()

            # postprocess
            det_xy, det_cls, instance_mask = u.nms_predicted_center(scan[0], scan_phi, pred_cls, pred_reg) # remove batch dimension, as batch size for evaluation is given as 1
            dets_xy.append(det_xy)
            dets_cls.append(det_cls)
            instance_masks.append(instance_mask)

            epe_batch, aae_batch = loss_fn_eval(pred_flow, target_flow)

            for i in range(test_loader.batch_size):
                pred_flow[i] = u.canonical_to_global_flow_torch(pred_flow[i], scan_phi)
                target_flow[i] = u.canonical_to_global_flow_torch(target_flow[i], scan_phi)

            if output_dir is not None:

                if scans is None:
                    scans = scan
                    odoms = odom
                    pred_flows = pred_flow.data.cpu().numpy()
                    target_flows = target_flow.data.cpu().numpy()
                    epe_list = epe_batch.data.cpu().numpy()
                    aae_list = aae_batch.data.cpu().numpy()

                else:
                    scans = np.vstack([scans, scan])
                    odoms = np.hstack([odoms, odom])
                    pred_flows = np.vstack([pred_flows, pred_flow.data.cpu().numpy()])
                    target_flows = np.vstack([target_flows, target_flow.data.cpu().numpy()])
                    epe_list = np.hstack([epe_list, epe_batch.data.cpu().numpy()])
                    aae_list = np.hstack([aae_list, aae_batch.data.cpu().numpy()])

            eval_loss += torch.mean(epe_batch).item()

        print("Eval loss: ", eval_loss / len(test_loader))

    dets_xy = np.array(dets_xy)
    dets_cls = np.array(dets_cls)
    instance_masks = np.array(instance_masks)

    if output_dir is not None:
        print("Plotting flow sequence")
        path = os.path.join(output_dir, "tmp_videos")
        os.makedirs(path, exist_ok=True)

        scan_phi = u.get_laser_phi()
        scans_xy = u.rphi_to_xy(scans, np.repeat(scan_phi.reshape(1, -1), len(scans), axis=0))
        scans_xy = np.stack(scans_xy, axis=-1)

        hsv_pred_flows, hsv_target_flows = [], []
        for i in range(len(pred_flows)):
            hsv_pred_flows.append(u.flow_to_hsv(pred_flows[i]))
            hsv_target_flows.append(u.flow_to_hsv(target_flows[i]))
        hsv_pred_flows = np.asarray(hsv_pred_flows)
        hsv_target_flows = np.asarray(hsv_target_flows)

        # arrow_path = os.path.join(path, "flow_pred_target_arrow.mp4")
        # print('Writing arrow sequence to {}'.format(arrow_path))
        # plot_sequence_gt_pred_arrow(scans_xy[..., 0], scans_xy[..., 1],
        #                             path=arrow_path,
        #                             pred=pred_flows,
        #                             target=target_flows,
        #                             epe_list=epe_list,
        #                             aae_list=aae_list)

        hsv_path = os.path.join(path, "flow_pred_target_hsv.mp4")
        print('Writing hsv sequence to {}'.format(hsv_path))
        # plot_sequence_gt_pred_hsv(scans_xy[..., 0], scans_xy[..., 1],
        #                           path=hsv_path,
        #                           pred=hsv_pred_flows,
        #                           target=hsv_target_flows,
        #                           epe_list=epe_list,
        #                           aae_list=aae_list)

        person_flow_path = os.path.join(path, "person_flow_hsv.mp4")
        plot_person_flow(scans,
                         dets_xy,
                         dets_cls,
                         instance_masks,
                         path=person_flow_path,
                         pred=hsv_pred_flows,
                         target=hsv_target_flows,
                         epe_list=None,
                         aae_list=None)

def eval_Bb_regression(model, test_loader, cfg, output_dir=None, tb_logger=None):
    reg = 1e-6
    eval_loss = 0.0
    model.eval()
    ious, dimensions_err, orientations_err = [], [], []

    with torch.no_grad():
        for cur_it, batch in enumerate(test_loader):
            # unpack data
            input, target = batch['input'], batch['target']
            det_center = batch['det_center']

            # Move data to GPU
            input = torch.from_numpy(input).cuda(non_blocking=True).float()
            target = torch.from_numpy(target).cuda(non_blocking=True).float()

            pred = model(input)
            # pred[pred[..., -1] > np.pi, -1] -= 2 * np.pi
            # pred[pred[..., -1] < -np.pi, -1] += 2 * np.pi
            loss = model.loss_fn(pred, target[:, 2:])
            eval_loss += loss

            pred = pred.data.cpu().numpy()
            target = target.data.cpu().numpy()
            input = input.data.cpu().numpy()

            pred[..., -1] *= np.pi
            target[..., -1] *= np.pi

            target[0, :2] = target[0, :2] + det_center
            input = input + det_center

            # box_pred = Box3d(np.array([pred[0, 0], pred[0, 1], -0.7]), np.array([pred[0, 2], pred[0, 3], 0.5]), pred[0, 4])
            # box_pred = Box3d(np.array([det_center[0, 0], det_center[0, 1], -0.7]), np.array([pred[0, 0], pred[0, 1], 0.5]), np.arccos(pred[0, 2])) # cosine as last channel
            # box_target = Box3d(np.array([target[0, 0], target[0, 1], -0.7]), np.array([target[0, 2], target[0, 3], 0.5]), np.arccos(target[0, 4]))
            box_pred = Box3d(np.array([det_center[0, 0], det_center[0, 1], -0.7]), np.array([pred[0, 0], pred[0, 1], 0.5]), pred[0, 2]) # rot_z as last channel
            box_target = Box3d(np.array([target[0, 0], target[0, 1], -0.7]), np.array([target[0, 2], target[0, 3], 0.5]), target[0, 4])

            iou = rotate_iou_gpu_eval(np.hstack((det_center, pred)), target)[0, 0]
            ious.append(iou)

            # position_err = np.linalg.norm(pred[0][:2] - target[0][:2])
            dimension_err = np.sum(np.abs(pred[0, :2] - target[0, 2:4]))
            # orientation_err = np.abs(np.arccos(pred[0, -1]) - np.arccos(target[0, -1]))  # cosine as last channel
            orientation_err = np.abs(pred[0][-1] - target[0][-1]) # rot_z as last channel
            dimensions_err.append(dimension_err)
            orientations_err.append(orientation_err)

            if cur_it < 2000 or cur_it > 2200:
                continue


            # ===============test===========

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)

            ax.cla()
            ax.set_title('Eval error. Dimension: {:0.3f} [m], Orientation: {:0.3f} [rad], IOU: {:0.3f}'.format(dimension_err, orientation_err, iou))
            ax.set_aspect("equal")
            # ax.set_xlim(-10, 10)
            # ax.set_ylim(-10, 10)

            #pred
            ax.scatter(input[..., 0], input[..., 1], s=1, c='blue')
            box_pred.draw_bev(ax, c='purple')
            c = plt.Circle(det_center[0], radius=0.01, color='green', fill=True) #perturbed center
            ax.add_artist(c)

            # target
            box_target.draw_bev(ax, c='red')
            c = plt.Circle(target[0, :2], radius=0.01, color='red', fill=True)
            ax.add_artist(c)

            plt.savefig("./tmp_imgs/eval_pointnet/frame_%04d.png" % cur_it)
            plt.close(fig)

            # ===============test===========

        print("Eval loss: ", eval_loss / len(test_loader))
        print("avg dimension error: {:0.3f} [m]".format(np.mean(dimensions_err)))
        print("avg orientation error: {:0.3f} [rad]".format(np.mean(orientations_err)))
        print("avg IOU: {:0.3f}".format(np.mean(ious)))

def model_fn_eval_box_reg(model, eval_loader):
    reg = 1e-6
    eval_loss = 0.0
    ious, dimensions_err, orientations_err = [], [], []
    model.eval()

    with torch.no_grad():
        for cur_it, batch in enumerate(eval_loader):
            # unpack data
            input, target = batch['input'], batch['target']
            det_center = batch['det_center']

            # Move data to GPU
            input = torch.from_numpy(input).cuda(non_blocking=True).float()
            target = torch.from_numpy(target).cuda(non_blocking=True).float()

            pred = model(input)
            # pred[pred[..., -1] > np.pi, -1] -= 2 * np.pi
            # pred[pred[..., -1] < -np.pi, -1] += 2 * np.pi
            loss = model.loss_fn(pred, target[:, 2:])

            pred = pred.data.cpu().numpy()
            target = target.data.cpu().numpy()
            pred[..., -1] *= np.pi
            target[..., -1] *= np.pi

            iou = rotate_iou_gpu_eval(np.hstack((np.zeros_like(det_center), pred)), target) # canonical case
            # iou = rotate_iou_gpu_eval(np.hstack((det_center, pred)), target) # global case
            ious.append(np.mean(np.diag(iou)))

            # position_err = np.linalg.norm(pred[0][:2] - target[0][:2], axis=1)
            dimension_err = np.sum(np.abs(pred[:, :2] - target[:, 2:4]), axis=1)
            # orientation_err = np.abs(np.arccos(pred[:, -1]) - np.arccos(target[:, -1])) # cosine as last channel
            orientation_err = np.abs(pred[:, -1] - target[:, -1]) # rot_z as last channel
            dimensions_err.append(np.mean(dimension_err))
            orientations_err.append(np.mean(orientation_err))

            eval_loss += loss

    return eval_loss / len(eval_loader), np.mean(dimensions_err), np.mean(orientations_err), np.mean(ious)

def eval_BB_reg_baseline(dataset):
    """
    Baseline evaluation. The prediction is the minimum horizontal (rot_z = 0) bounding box that wraps the input points.
    The IOU of such prediction and target is calculate to establish baseline performance
    :param dataset: The jrdb dataset for training bounding box regression in the project
    :return: no return value
    """
    eval_loss = 0.0
    ious, dimensions_err, orientations_err = [], [], []

    l_mean = np.mean(np.array(dataset.targets)[..., 2])
    w_mean = np.mean(np.array(dataset.targets)[..., 3])

    for cur_it, (input, target, det_center) in enumerate(zip(dataset.inputs, dataset.targets, dataset.dets_center)):

        # 4 corner points of the bounding box
        input -= det_center
        top_left = np.array([np.min(input[..., 0]), np.max(input[..., 1])])
        top_right = np.array([np.max(input[..., 0]), np.max(input[..., 1])])
        bottom_left = np.array([np.min(input[..., 0]), np.min(input[..., 1])])
        bottom_right = np.array([np.max(input[..., 0]), np.min(input[..., 1])])

        box_vert = np.array([top_left, top_right, bottom_left, bottom_right])
        # length = np.max(input[..., 0]) - np.min(input[..., 0])
        # width = np.max(input[..., 1]) - np.min(input[..., 1])
        length = 2 * np.max(np.abs(input[..., 0]))
        width = 2 * np.max(np.abs(input[..., 1]))

        # pred = np.array([length, width, 0])
        pred = np.array([l_mean, w_mean, 0.5 * np.pi])
        input += det_center

        box_pred = Box3d(np.array([det_center[0], det_center[1], -0.7]), np.array([pred[0], pred[1], 0.5]), pred[2])  # rot_z as last channel
        box_target = Box3d(np.array([target[0], target[1], -0.7]), np.array([target[2], target[3], 0.5]), target[4])

        iou = rotate_iou_gpu_eval(np.hstack((det_center, pred)).reshape(1, -1), target.reshape(1, -1))[0, 0]
        ious.append(iou)

        # position_err = np.linalg.norm(pred[0][:2] - target[0][:2], axis=1)
        dimension_err = np.sum(np.abs(pred[:2] - target[2:4]))
        # orientation_err = np.abs(np.arccos(pred[:, -1]) - np.arccos(target[:, -1]))  # cosine as last channel
        orientation_err = np.abs(pred[-1] - target[-1]) # rot_z as last channel
        dimensions_err.append(np.mean(dimension_err))
        orientations_err.append(np.mean(orientation_err))

        if cur_it > 200:
            continue

        # ===============test===========

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.cla()
        ax.set_title('Eval error. Dimension: {:0.3f} [m], Orientation: {:0.3f} [rad], IOU: {:0.3f}'.format(dimension_err, orientation_err, iou))
        ax.set_aspect("equal")
        # ax.set_xlim(-10, 10)
        # ax.set_ylim(-10, 10)

        #pred
        ax.scatter(input[..., 0], input[..., 1], s=1, c='blue')
        box_pred.draw_bev(ax, c='purple')
        c = plt.Circle(det_center, radius=0.01, color='green', fill=True) #perturbed center
        ax.add_artist(c)

        # target
        box_target.draw_bev(ax, c='red')
        c = plt.Circle(target[:2], radius=0.01, color='red', fill=True)
        ax.add_artist(c)

        plt.savefig("./tmp_imgs/eval_baseline/frame_%04d.png" % cur_it)
        plt.close(fig)

        # ===============test===========

    print("Eval loss: ", np.mean(dimensions_err) + np.mean(orientations_err))
    print("avg dimension error: {:0.3f} [m]".format(np.mean(dimensions_err)))
    print("avg orientation error: {:0.3f} [rad]".format(np.mean(orientations_err)))
    print("avg IOU: {:0.3f}".format(np.mean(ious)))



def compute_iou_aabb(box1, box2):
    box1_vertices = box1.to_vertices()[:, [2, 0]] # get top left and bottom right corner vertices of the box
    box2_vertices = box2.to_vertices()[:, [2, 0]]
    cs1, ss1 = np.cos(box1.rot_z + np.pi), np.sin(box1.rot_z + np.pi)
    cs2, ss2 = np.cos(box2.rot_z + np.pi), np.sin(box2.rot_z + np.pi)
    R1 = np.array([[cs1, ss1, 0], [-ss1, cs1, 0], [0, 0, 1]], dtype=np.float32)
    R2 = np.array([[cs2, ss2, 0], [-ss2, cs2, 0], [0, 0, 1]], dtype=np.float32)
    box1_vertices = (R1.T @ (box1_vertices - box1.xyz) + box1.xyz)[:2] # transform box back to axis aligned box
    box2_vertices = (R2.T @ (box2_vertices - box2.xyz) + box2.xyz)[:2]

    box1_area = (box1_vertices[0, 1] - box1_vertices[0, 0]) * (box1_vertices[1, 0] - box1_vertices[1, 1])
    box2_area = (box2_vertices[0, 1] - box2_vertices[0, 0]) * (box2_vertices[1, 0] - box2_vertices[1, 1])

    box_inter_vertices = np.array([[max(box1_vertices[0, 0], box2_vertices[0, 0]), min(box1_vertices[0, 1], box2_vertices[0, 1])],
                                   [min(box1_vertices[1, 0], box2_vertices[1, 0]), max(box1_vertices[1, 1], box2_vertices[1, 1])]])

    # compute area of intersection
    area_inter = max(0, box_inter_vertices[0, 1] - box_inter_vertices[0, 0]) * max(0, box_inter_vertices[1, 0] - box_inter_vertices[1, 1])
    # compute area of union
    area_union = box1_area + box2_area - area_inter

    return area_inter / area_union

