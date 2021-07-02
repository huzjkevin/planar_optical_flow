import torch
import os
import argparse
import yaml
from shutil import copyfile

from src.depracted.model import FlowDROW_pretrained
from src.utils.viz_utils import *
import src.utils.utils as u
import numpy as np

def _plot_scanner_icon(ax, odo_xy, odo_rot):
    # sample points along a half circle
    scan_phi = u.get_laser_phi()
    rad_tmp = 0.5 * np.ones(len(scan_phi), dtype=np.float)
    xy_scanner = u.rphi_to_xy(rad_tmp, scan_phi)
    xy_scanner = np.stack(xy_scanner, axis=1)

    # transform points to target frame
    xy_scanner_rot = np.matmul(xy_scanner, odo_rot.T) + odo_xy

    # plot
    ax.plot(xy_scanner_rot[:, 0], xy_scanner_rot[:, 1], c='black')
    ax.plot((odo_xy[0], xy_scanner_rot[0, 0] * 1.0), (odo_xy[1], xy_scanner_rot[0, 1] * 1.0), c='black')
    ax.plot((odo_xy[0], xy_scanner_rot[-1, 0] * 1.0), (odo_xy[1], xy_scanner_rot[-1, 1] * 1.0), c='black')



torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--cfg", type=str, required=True, help="configuration of the experiment")
parser.add_argument("--ckpt", type=str, required=False, default=None)
args = parser.parse_args()

with open(args.cfg, 'r') as f:
    cfg = yaml.safe_load(f)
    cfg['name'] = os.path.basename(args.cfg).split(".")[0] + cfg['tag']

if __name__ == "__main__":

    # Create directory for storing results
    root_result_dir = os.path.join('../', 'output', cfg['name'])
    os.makedirs(root_result_dir, exist_ok=True)
    copyfile(args.cfg, os.path.join(root_result_dir, os.path.basename(args.cfg)))

    ckpt_dir = os.path.join(root_result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    print('Loading data')
    sample = 1000
    seq_name = './data/DROWv2-data/test/run_t_2015-11-26-11-22-03.bag'
    # seq_name = './data/DROWv2-data/train/run_2015-11-26-15-52-55-l.bag'
    scans_data = np.genfromtxt(seq_name + '.csv', delimiter=',')[:sample]
    scans_t = scans_data[:, 1]
    scans = scans_data[:, 2:]
    scan_phi = u.get_laser_phi()

    # odometry, used only for plotting
    odometry = np.genfromtxt(seq_name + '.odom2', delimiter=',')[:sample]
    odoms_t = odometry[:, 1]
    odoms = odometry[:, 2:]
    odoms_phi = odometry[:, -1]

    print("Prepare model")
    model = FlowDROW_pretrained(num_scans=cfg['num_scans'],
                            num_pts=cfg['cutout_kwargs']['num_cutout_pts'],
                            focal_loss_gamma=cfg['focal_loss_gamma'],
                            alpha=cfg['similarity_kwargs']['alpha'],
                            window_size=cfg['similarity_kwargs']['window_size'],
                            pedestrian_only=cfg['pedestrian_only'])

    model.cuda()

    cur_ckpt = "{}.pth".format(os.path.join(ckpt_dir, "ckpt_e{}".format(cfg["epochs"])))
    model.load_state_dict(torch.load(cur_ckpt)["model_state"])
    model.eval()

    # scanner location
    rad_tmp = 0.5 * np.ones(len(scan_phi), dtype=np.float)
    xy_scanner = u.rphi_to_xy(rad_tmp, scan_phi)
    xy_scanner = np.stack(xy_scanner, axis=1)

    # plot
    gs = GridSpec(1, 2, height_ratios=[1.0], width_ratios=[1.0, 0.0])
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(gs[0])
    canvas = FigureCanvas(fig)
    w, h = canvas.get_width_height()
    result_path = os.path.join(root_result_dir, '../tmp_videos')
    os.makedirs(result_path, exist_ok=True)
    fname = os.path.join(result_path, 'person_flow.avi')
    writer = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w, h))

    print('Start inference')
    reg = 1e-2
    cls_thresh = 0.5
    odom_idx = 0
    prev_odom_idx = -1

    for i in range(len(scans)):
        plt.cla()

        ax.set_aspect('equal')
        ax.set_xlim(50, 80)
        ax.set_ylim(85, 115)

        ax.set_title('Frame: %s' % i)
        # ax.axis("off")

        # find matching odometry
        while odom_idx < len(odoms_t) - 1 and odoms_t[odom_idx] < scans_t[i]:
            odom_idx += 1
        odom_phi = odoms_phi[odom_idx]
        odom1 = odoms[odom_idx]

        odom_rot = u._phi_to_rotation_matrix(odom_phi)

        # plot scanner icon
        _plot_scanner_icon(ax, odom1[:2], odom_rot)

        # inference
        scan = scans[i]
        scan_xy = np.array(u.rphi_to_xy(scan, scan_phi)).T
        scan_xy_world = np.matmul(scan_xy, odom_rot.T) + odom1[:2].reshape(1, 2)
        # ax.scatter(scan_xy_world[:, 0], scan_xy_world[:, 1], s=1, c='blue')

        input = u.scans_to_cutout(scan[None, ...], scan_phi, stride=1, **cfg['cutout_kwargs'])
        input = torch.from_numpy(input).cuda().float()

        pred_cls, pred_reg, pred_flow = model(input.unsqueeze(dim=0),
                                              torch.from_numpy(scan).cuda(non_blocking=True).float().view(1, -1),
                                              testing=True)
        pred_cls = torch.sigmoid(pred_cls[0]).data.cpu().numpy()
        pred_reg = pred_reg[0].data.cpu().numpy()
        pred_flow = u.canonical_to_global_flow_torch(pred_flow[0], scan_phi).data.cpu().numpy()

        #Post-processing
        dets_xy, dets_cls, instance_mask = u.nms_predicted_center(scan, scan_phi, pred_cls, pred_reg)

        # plot detection and person flow in world frame
        if prev_odom_idx > 0:
            odom0 = odoms[prev_odom_idx]
            # Transform dets and flow to world frame
            dets_xy_world = np.matmul(dets_xy, odom_rot.T) + odom1[:2].reshape(1, 2)
            pred_flow_world = np.matmul(pred_flow, odom_rot.T) + (odom1 - odom0)[:2].reshape(1, 2)  # TODO: Make sure if adding translation is necessary
            pred_flow_world_hsv = u.flow_to_hsv(pred_flow_world)  # HSV encode
            for j in range(len(dets_xy)):
                instance_ids = instance_mask == j + 1
                if dets_cls[j] < cls_thresh:
                    ax.scatter(scan_xy_world[instance_ids, 0], scan_xy_world[instance_ids, 1], s=3, c='black')
                else:
                    ax.quiver(dets_xy_world[j][0], dets_xy_world[j][1],
                              np.mean(pred_flow_world[instance_ids, 0]),
                              np.mean(pred_flow_world[instance_ids, 1]),
                              color=np.mean(pred_flow_world_hsv[instance_ids], axis=0),
                              linewidth=0.1)

                    ax.scatter(scan_xy_world[instance_ids, 0], scan_xy_world[instance_ids, 1], s=3, c=pred_flow_world_hsv[instance_ids])
                    c = plt.Circle(dets_xy_world[j], radius=0.5, color='r', fill=False)
                    ax.add_artist(c)

        # plot target flow
        # if prev_odom_idx > 0:
        #     odom0 = odoms[prev_odom_idx]
        #     disp = u.get_displacement_from_odometry(scan_xy, odom0, odom1)
        #     disp_world = np.matmul(disp, odom_rot.T)
        #
        #     # odom_dt = max(odoms_t[odom_idx] - odoms_t[prev_odom_idx], reg)
        #
        #     non_zero_mask = np.hypot(disp_world[:, 0], disp_world[:, 1]) > 0
        #     arrow_xy = scan_xy_world[non_zero_mask]
        #     arrow_dxy = disp_world[non_zero_mask]
        #
        #     for arrow_idx in range(0, np.sum(non_zero_mask), 5):
        #         ax.arrow(arrow_xy[arrow_idx, 0], arrow_xy[arrow_idx, 1],
        #                   arrow_dxy[arrow_idx, 0], arrow_dxy[arrow_idx, 1])

        prev_odom_idx = odom_idx

        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, -1)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    plt.close(fig)
    writer.release()