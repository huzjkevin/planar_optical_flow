import sys
# sys.path.append("./..")
# sys.path.append("/home/hu/Projects/planar_optical_flow")

import torch
import os
import argparse
import yaml
from shutil import copyfile

from src.utils.dataset_dr_spaam import create_test_dataloader, create_dataloader
from src.depracted.model import SpatialDROW, FlowDROW_pretrained
from src.utils.viz_utils import *
import src.utils.utils as u
import numpy as np

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--cfg", type=str, required=True, help="configuration of the experiment")
parser.add_argument("--ckpt", type=str, required=False, default=None)
args = parser.parse_args()

with open(args.cfg, 'r') as f:
    cfg = yaml.safe_load(f)
    cfg['name'] = os.path.basename(args.cfg).split(".")[0] + cfg['tag']

if __name__ == "__main__":
    #Create directory for storing results
    root_result_dir = os.path.join('../', 'output_cluster', cfg['name'])
    os.makedirs(root_result_dir, exist_ok=True)
    copyfile(args.cfg, os.path.join(root_result_dir, os.path.basename(args.cfg)))

    ckpt_dir = os.path.join(root_result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Prepare data")
    test_loader = create_test_dataloader(data_path="../data/DROWv2-data",
                                         num_scans=cfg['num_scans'],
                                         network_type=cfg['network'],
                                         cutout_kwargs=cfg['cutout_kwargs'],
                                         polar_grid_kwargs=cfg['polar_grid_kwargs'],
                                         pedestrian_only=cfg['pedestrian_only'],
                                         split='train')

    #Prepare model
    print("Prepare model")
    model = FlowDROW_pretrained(num_scans=cfg['num_scans'],
                                num_pts=cfg['cutout_kwargs']['num_cutout_pts'],
                                focal_loss_gamma=cfg['focal_loss_gamma'],
                                alpha=cfg['similarity_kwargs']['alpha'],
                                window_size=cfg['similarity_kwargs']['window_size'],
                                pedestrian_only=cfg['pedestrian_only'])

    model.cuda()

    #Load trained model
    # cur_ckpt = "{}.pth".format(os.path.join(ckpt_dir, "ckpt_e{}".format(cfg["epochs"])))
    cur_ckpt = "{}.pth".format(os.path.join(ckpt_dir, "ckpt_e{}".format(200)))
    model.load_state_dict(torch.load(cur_ckpt)["model_state"])
    model.eval()

    #Testing
    print("Start inference")
    reg = 1e-5
    eval_loss = 0.0
    model.eval()
    scan_phi = u.get_laser_phi()
    dets_xy = []
    dets_cls = []
    instance_masks = []

    scans = None
    pred_flows = None
    scan_odoms = None
    target_flows = None

    with torch.no_grad():
        for cur_it, batch in enumerate(test_loader):
            scan = batch['scans'][:, -2]
            input = batch['input']
            odom_t = batch['odom_t']
            odom = batch['odom']
            scan_odom = batch['scan_odom']
            target_flow = batch['target_flow']

            input = torch.from_numpy(input).cuda(non_blocking=True).float()

            pred_cls, pred_reg, pred_flow = model(input, testing=True)

            pred_cls = torch.sigmoid(pred_cls[0]).data.cpu().numpy()
            pred_reg = pred_reg[0].data.cpu().numpy()

            # postprocess
            det_xy, det_cls, instance_mask = u.nms_predicted_center(scan[0], scan_phi, pred_cls, pred_reg)  # remove batch dimension, as batch size for evaluation is given as 1
            dets_xy.append(det_xy)
            dets_cls.append(det_cls)
            instance_masks.append(instance_mask)

            odom_t += reg

            for i in range(test_loader.batch_size):
                pred_flow[i] = u.canonical_to_global_flow_torch(pred_flow[i], scan_phi)
                pred_flow[i] = (u.canonical_to_global_flow_torch(pred_flow[i], np.ones_like(scan_phi) * (scan_odom[0, -1])) + torch.from_numpy(scan_odom[0][:-1]).cuda(non_blocking=True).float()) / odom_t[0]

                # target_flow[i] = u.canonical_to_global_flow(target_flow[i], scan_phi)
                # target_flow[i] = u.canonical_to_global_flow(target_flow[i], np.ones_like(scan_phi) * (scan_odom[0, -1])) + odom[0, :-1]

            if scans is None:
                scans = scan
                scan_odoms = scan_odom
                pred_flows = pred_flow.data.cpu().numpy()
                target_flows = target_flow

            else:
                scans = np.vstack([scans, scan])
                scan_odoms = np.vstack([scan_odoms, scan_odom])
                pred_flows = np.vstack([pred_flows, pred_flow.data.cpu().numpy()])
                target_flows = np.vstack([target_flows, target_flow])

    dets_xy = np.array(dets_xy)
    dets_cls = np.array(dets_cls)
    instance_masks = np.array(instance_masks)

    print("Plotting flow sequence")
    path = os.path.join(root_result_dir, "../tmp_videos")
    os.makedirs(path, exist_ok=True)

    scan_phi = u.get_laser_phi()
    scans_xy = u.rphi_to_xy(scans, np.repeat(scan_phi.reshape(1, -1), len(scans), axis=0))
    scans_xy = np.stack(scans_xy, axis=-1)

    hsv_pred_flows, hsv_target_flows = [], []
    for i in range(len(pred_flows)):
        hsv_pred_flows.append(u.flow_to_hsv(pred_flows[i]))
    hsv_pred_flows = np.asarray(hsv_pred_flows)

    sample = 1000
    # person_flow_path = os.path.join(path, "det_person_flow_arrow_world.avi")
    # plot_person_flow_fixed_pose(scans[:sample],
    #                             dets_xy[:sample],
    #                             dets_cls[:sample],
    #                             instance_masks[:sample],
    #                             path=person_flow_path,
    #                             odoms_phi=scan_odoms[:sample][..., -1],
    #                             pred_hsv=hsv_pred_flows[:sample],
    #                             pred_arrow=pred_flows[:sample])

    target_flow_path = os.path.join(path, "target_flow_arrow_world.avi")
    plot_sequence(scans[:sample], np.repeat(scan_phi.reshape(1, -1), len(scans[:sample]), axis=0),
                path=target_flow_path,
                # odoms_phi=odoms[:sample][..., -1],
                arrow=target_flows[:sample])