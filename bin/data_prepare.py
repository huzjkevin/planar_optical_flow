import sys
sys.path.append("/home/hu/Projects/planar_optical_flow")
# print(sys.path)
from glob import glob
import os
import numpy as np
from src.utils.utils import *
from src.utils.viz_utils import *
import numpy as np
import cv2 as cv
import colorsys as color
import pandas as pd


def load_odom(fname):  # --Kevin
    odoms = np.genfromtxt(fname, delimiter=",", dtype=[('seq', 'u4'), ('t', 'f4'), ('xya', 'f4', 3)])
    odom_ns = odoms["seq"]
    odom_t = odoms["t"]
    odoms = odoms["xya"]
    return odom_ns, odom_t, odoms

def load_odom_file(fname):
    odoms = np.genfromtxt(fname + '.difodom', delimiter=',')
    odoms_t = odoms[:, 0]
    odoms = odoms[:, 1:]

    return odoms_t, odoms

def get_flow_target(scan, scan_phi, odom_t, odom): #--Kevin
    #scan = (450,), scan_phi = (450,)
    #odom_t = (1,), odom = (3,)
    reg = 1e-6
    v_odom = odom[:2] / (odom_t + reg)
    w_odom = np.asarray([0, 0, odom[-1] / (odom_t + reg)])
    # scan_x, scan_y = np.array(rphi_to_xy(scan, scan_phi))
    scan_xy = np.array(rphi_to_xy(scan, scan_phi)).T
    scan_3d = np.hstack([scan_xy, np.zeros((len(scan_xy), 1))])

    v_linear = v_odom
    v_angular = w_odom
    v_rot = np.cross(v_angular, scan_3d)[:, :2]
    v_scan = v_rot + v_linear
    d_scan = v_scan * odom_t

    # err = np.abs(w_odom[-1]) * scan - np.linalg.norm(v_rot, axis=1)

    return d_scan

def flow_to_hsv(flow):
    num_pts, _ = flow.shape

    hsv = np.zeros((num_pts, 1, 3)).astype("uint8")
    hsv[..., 1] = 255
    # r, phi = xy_to_rphi(flow[..., 0], flow[..., 1])
    r, phi = cv.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = phi * 360 / np.pi / 2
    hsv[..., 2] = cv.normalize(r, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB).reshape(-1, 3)
    return rgb

def load_scan_file(seq_name):
    data = np.genfromtxt(seq_name + '.csv', delimiter=",")
    seqs = data[:, 0].astype(np.uint32)
    times = data[:, 1].astype(np.float32)
    scans = data[:, 2:].astype(np.float32)
    return seqs, times, scans

def load_flow_file(seq_name, shape=(-1, 450, 2)):
    flow_targets = np.genfromtxt(seq_name + '.flow', delimiter=',')

    return flow_targets.reshape(shape)

data_path="./../data/DROWv2-data"
split = "test"
seq_names = [f[:-4] for f in glob(os.path.join(data_path, split, '*.csv'))]
os.makedirs(os.path.join(data_path, split), exist_ok=True)

print("Loading odom files\n")

print("Compute odom difference\n")
for f in seq_names:
    odom_ns, odom_t, odom = load_odom(f + ".odom2")
    diff_odom_t = np.concatenate((odom_t[1:] - odom_t[:-1], [0]))
    diff_odom = np.concatenate((odom[1:] - odom[:-1], [[0] * 3]))

    test = np.hstack([diff_odom_t.reshape(-1, 1), diff_odom])
    print("Save odom differnece to: {}".format(f + ".difodom"))
    np.savetxt(f + '.difodom', test, fmt="%8.6f", delimiter=',')
print("{} files saved\n".format(len(seq_names)))

print("Loading scan files\n")
scans_ns, scans_t, scans = zip(*[load_scan_file(f) for f in seq_names])
odoms_t, odoms = zip(*[load_odom_file(f) for f in seq_names])

scan_phi = get_laser_phi()

# scan_path = "/media/kevin/Kevin/scan_videos"
# os.makedirs(scan_path, exist_ok=True)
# a = len(scans)
# for i in range(len(scans)):
#     fname = os.path.join(scan_path, "scan{}.avi".format(i))
#     plot_sequence(scans[i], np.repeat(scan_phi.reshape(1, -1), len(scans[i]), axis=0), path=fname)
#
print("Compute flow targets\n")
for idx, fname in enumerate(seq_names):
    flow_targets = []
    for scan, odom_t, odom in zip(scans[idx], odoms_t[idx], odoms[idx]):
        flow_target = get_flow_target(scan, scan_phi, odom_t, odom)
        flow_targets.append(flow_target)

    flow_targets = np.asarray(flow_targets)
    print("Save flow target to: {}".format(fname + ".flow"))
    np.savetxt(fname + ".flow", flow_targets.reshape(-1, 450 * 2), fmt="%10.8f", delimiter=',')
print("{} files saved\n".format(len(seq_names)))
print("Finished")
#
# flow_targets = [load_flow_file(f, shape=(-1, 450, 2)) for f in seq_names]
#
# scans_sample = scans[0][500:1000]
# flow_sample = flow_targets[0][500:1000]
#
# flow_arrow_path = './../tmp_videos/flow_arrow_samples_test.avi'
# plot_sequence(scans_sample, np.repeat(scan_phi.reshape(1, -1), len(scans_sample), axis=0), path=flow_arrow_path, arrow=flow_sample)
#
# flow_hsv = "./../tmp_videos/flow_hsv_samples_test.avi"
# flow_hsv_opencv = "./../tmp_videos/flow_hsv_samples_opencv.avi"
# scans_sample,
# colors = []
# for flow in flow_sample:
#     hsv = flow_to_hsv(flow)
#     colors.append(hsv)
# colors = np.asarray(colors)
# plot_sequence(scans_sample, np.repeat(scan_phi.reshape(1, -1), len(scans_sample), axis=0), path=flow_hsv, color=colors)

# non_zero_ids = [np.all(np.all(flow == 0.0, axis=1) == 0.0, axis=1) for flow in flow_targets]
#
# scans_ns = list(scans_ns)
# scans_t = list(scans_t)
# scans = list(scans)
# dets_ns = list(dets_ns)
# dets_wc = list(dets_wc)
# dets_wa = list(dets_wa)
# dets_wp = list(dets_wp)
# odoms_t = list(odoms_t)
# odoms = list(odoms)
# scan_odoms_t = list(scan_odoms_t)
# scan_odoms = list(scan_odoms)
#
# self.seq_names = []
#
# self.scans_ns = []
# self.scans_t = []
# self.scans = []
# self.scans_next = []
# self.flow_targets = []
#
# self.dets_ns = []
# self.dets_wc = []
# self.dets_wa = []
# self.dets_wp = []
# self.odoms_t = []
# self.odoms = []
# self.scan_odoms_t = []
# self.scan_odoms = []
#
# for seq_idx, indices in enumerate(non_zero_ids):
#     if np.any(indices):
#         self.seq_names.append(seq_names[seq_idx])
#         self.scans_ns.append(scans_ns[seq_idx][indices])
#         self.scans_t.append(scans_t[seq_idx][indices])
#         self.scans.append(scans[seq_idx][indices])
#         self.scans_next.append(scans_next[seq_idx][indices])
#         self.flow_targets.append(flow_targets[seq_idx][indices])
#
#         self.dets_ns.append(dets_ns[seq_idx])
#         self.dets_wc.append(dets_wc[seq_idx])
#         self.dets_wa.append(dets_wa[seq_idx])
#         self.dets_wp.append(dets_wp[seq_idx])
#         self.odoms_t.append(odoms_t[seq_idx][indices])
#         self.odoms.append(odoms[seq_idx][indices])
#         self.scan_odoms_t.append(scan_odoms_t[seq_idx][indices])
#         self.scan_odoms.append(scan_odoms[seq_idx][indices])