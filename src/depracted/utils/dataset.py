from glob import glob
import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
# import utils as u
# import viz_utils as vu
from . import utils as u
from . import viz_utils as vu

def create_dataloader(data_path, num_scans, batch_size, num_workers, network_type="prototype",
                      train_with_val=False, use_data_augumentation=False):
    pass

def create_test_dataloader(data_path, num_scans, batch_size, num_workers, network_type="prototype",
                      train_with_val=False, use_data_augumentation=False):
    pass

class FlowDataset(Dataset):
    def __init__(self, data_path, split='train', testing=False, train_with_val=False):

        if train_with_val:
            seq_names = [f[:-4] for f in glob(os.path.join(data_path, split, '*.csv'))]
            seq_names += [f[:-4] for f in glob(os.path.join(data_path, 'val', '*.csv'))]
        else:
            seq_names = [f[:-4] for f in glob(os.path.join(data_path, split, '*.csv'))]

        # if testing:
        #     self.seq_names = seq_names[-3:]
        # else:
        #     self.seq_names = seq_names[:3]
        self.seq_names = seq_names[:5]

        # Pre-load scans, odometry and flow target
        self.scans_ns, self.scans_t, self.scans = zip(*[self._load_scan_file(f) for f in self.seq_names]) #n_seqs * n_scans * n_pts
        self.scans_next = [np.vstack([s[1:], s[-1].reshape(1, -1)]) for s in self.scans]
        self.scans_ns = np.hstack(self.scans_ns)
        self.scans_t = np.hstack(self.scans_t)
        self.scans = np.vstack(self.scans)
        self.scans_next = np.vstack(self.scans_next)

        self.odoms_t, self.odoms = zip(*[self._load_odom_file(f) for f in self.seq_names])
        self.odoms_t = np.hstack(self.odoms_t)
        self.odoms = np.vstack(self.odoms)

        _, _, scan_odoms = zip(*[self._load_odom(f) for f in self.seq_names])
        scan_odoms = np.vstack(scan_odoms)
        self.scan_dir = scan_odoms[..., -1]

        n_pts = self.scans.shape[-1]
        self.flow_targets = [self._load_flow_file(f, shape=(-1, n_pts, 2)) for f in self.seq_names]
        self.flow_targets = np.vstack(self.flow_targets)

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        rtn_dict = {}
        # rtn_dict['seq_name'] = self.seq_names[idx]

        # Scan
        scan = self.scans[idx]
        scan_next = self.scans_next[idx]

        # Angle
        scan_phi = u.get_laser_phi()
        rtn_dict['phi_grid'] = scan_phi

        # Odometry. Note that here it refers to difference of odometry as individual odom is not meaningful
        odom_t = self.odoms_t[idx]
        odom = self.odoms[idx]

        # Flow target
        flow_target = self.flow_targets[idx]

        #Scan transformation
        scan_dir = self.scan_dir[idx]
        scan_xy = u.rphi_to_xy(scan, scan_phi)
        scan_xy_next = u.rphi_to_xy(scan_next, scan_phi)
        scan_xy = np.stack(scan_xy, axis=1)
        scan_xy_next = np.stack(scan_xy_next, axis=1)

        rot = np.array([[np.cos(odom[-1]), np.sin(odom[-1])],
                        [-np.sin(odom[-1]), np.cos(odom[-1])]], dtype=np.float32)

        rot_trans = np.array([[np.cos(scan_dir), -np.sin(scan_dir)],
                              [np.sin(scan_dir), np.cos(scan_dir)]], dtype=np.float32)

        trans = np.matmul(odom[:-1], rot_trans.T)

        scan_xy_next_rot = np.matmul(scan_xy_next, rot.T) + trans

        rtn_dict['scan_pair'] = [scan_xy, scan_xy_next_rot]
        rtn_dict['odom_t'] = odom_t
        rtn_dict['odom'] = odom
        rtn_dict["flow_target"] = flow_target

        return rtn_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            if k in ["scan_pair", "flow_target"]:
                rtn_dict[k] = np.stack([sample[k] for sample in batch], axis=0)
            else:
                rtn_dict[k] = [sample[k] for sample in batch]

        return rtn_dict

    def _load_scan_file(self, seq_name):
        data = np.genfromtxt(seq_name + '.csv', delimiter=",")
        seqs = data[:, 0].astype(np.uint32)
        times = data[:, 1].astype(np.float32)
        scans = data[:, 2:].astype(np.float32)
        return seqs, times, scans

    def _load_odom(self, seq_name):  # --Kevin
        odoms = np.genfromtxt(seq_name + ".odom2", delimiter=",", dtype=[('seq', 'u4'), ('t', 'f4'), ('xya', 'f4', 3)])
        odom_ns = odoms["seq"]
        odom_t = odoms["t"]
        odoms = odoms["xya"]
        return odom_ns, odom_t, odoms

    def _load_odom_file(self, seq_name):
        odoms = np.genfromtxt(seq_name + '.difodom', delimiter=',')
        odoms_t = odoms[:, 0]
        odoms = odoms[:, 1:]

        return odoms_t, odoms

    def _load_flow_file(self, seq_name, shape=(-1, 450, 2)):
        flow_targets = np.genfromtxt(seq_name + '.flow', delimiter=',')

        return flow_targets.reshape(shape)

    def _scan_odom_matching(self, seq_idx, scans_t):
        odoms_ns, odoms_t, odoms = [], [], []
        diff_odoms_t, diff_odoms = [], []
        for t in scans_t:
            idx = np.argmin(np.abs(self.odoms_t[seq_idx] - t))
            odom_ns = self.odoms_ns[seq_idx][idx]
            odom_t = self.odoms_t[seq_idx][idx]
            odom = self.odoms[seq_idx][idx]
            diff_odom_t = self.diff_odoms_t[seq_idx][idx]
            diff_odom = self.diff_odoms[seq_idx][idx]
            odoms_ns.append(odom_ns)
            odoms_t.append(odom_t)
            odoms.append(odom)
            diff_odoms_t.append(diff_odom_t)
            diff_odoms.append(diff_odom)

        return odoms_ns, odoms_t, odoms, diff_odoms_t, diff_odoms

# The dataset that remove all static sequences
class FlowDatasetTmp(Dataset):
    def __init__(self, data_path, split='train', testing=False, train_with_val=False):

        if train_with_val:
            seq_names = [f[:-4] for f in glob(os.path.join(data_path, split, '*.csv'))]
            seq_names += [f[:-4] for f in glob(os.path.join(data_path, 'val', '*.csv'))]
        else:
            seq_names = [f[:-4] for f in glob(os.path.join(data_path, split, '*.csv'))]

        seq_names = seq_names[:5]

        if testing:
            seq_names = seq_names[:5]

        flow_targets = [self._load_flow_file(f, shape=(-1, 450, 2)) for f in seq_names]
        non_zero_ids = [np.all(np.all(flow == 0.0, axis=1) == 0.0, axis=1) for flow in flow_targets]

        self.seq_names = []
        self.flow_targets = []
        for seq_idx, indices in enumerate(non_zero_ids):
            if np.any(indices):
                self.seq_names.append(seq_names[seq_idx])
                self.flow_targets.append(flow_targets[seq_idx])

        if len(self.seq_names) == 0:
            print("No valid data")
            return

        print("{} valid files found".format(len(self.seq_names)))

        # Pre-load scans, annotation, odometry and flow target
        self.scans_ns, self.scans_t, self.scans = zip(*[self._load_scan_file(f) for f in self.seq_names]) #n_seqs * n_scans * n_pts
        self.scans_next = [np.vstack([s[1:], s[-1].reshape(1, -1)]) for s in self.scans]
        self.dets_ns, self.dets_wc, self.dets_wa, self.dets_wp = zip(*map(lambda f: self._load_det_file(f), self.seq_names))

        self.odoms_t, self.odoms = zip(*[self._load_odom_file(f) for f in self.seq_names])
        _, self.scan_odoms_t, self.scan_odoms = zip(*[self._load_odom(f) for f in self.seq_names])
        # self.scan_dir = scan_odoms[:, -1]

        # n_pts = self.scans.shape[-1]


        # Pre-compute mappings from detection index to scan index
        # such that idet2iscan[seq_idx][det_idx] = scan_idx
        self.idet2iscan = [{i: np.where(ss == d)[0][0] for i, d in enumerate(ds)}
                           for ss, ds in zip(self.scans_ns, self.dets_ns)]

        # Look-up list for sequence indices and annotation indices.
        self.flat_seq_inds, self.flat_det_inds = [], []
        for seq_idx, det_ns in enumerate(self.dets_ns):
            num_samples = len(det_ns)
            self.flat_seq_inds += [seq_idx] * num_samples
            self.flat_det_inds += range(num_samples)

    def __len__(self):
        return len(self.flat_det_inds)

    def __getitem__(self, idx):
        # idx = 10
        seq_idx = self.flat_seq_inds[idx]
        det_idx = self.flat_det_inds[idx]
        dets_ns = self.dets_ns[seq_idx][det_idx]

        rtn_dict = {}
        rtn_dict['seq_name'] = self.seq_names[seq_idx]
        rtn_dict['dets_ns'] = dets_ns

        # Annotation
        rtn_dict['dets_wc'] = self.dets_wc[seq_idx][det_idx]
        rtn_dict['dets_wa'] = self.dets_wa[seq_idx][det_idx]
        rtn_dict['dets_wp'] = self.dets_wp[seq_idx][det_idx]

        # Scan
        scan_idx = self.idet2iscan[seq_idx][det_idx]
        scan = self.scans[seq_idx][scan_idx]
        scan_next = self.scans_next[seq_idx][scan_idx]
        scan_t = self.scans_t[seq_idx][scan_idx]
        odom_idx = np.argmin(np.abs(self.scan_odoms_t[seq_idx] - scan_t))

        # Angle
        scan_phi = u.get_laser_phi()
        rtn_dict['phi_grid'] = scan_phi

        # Odometry. Note that here it refers to difference of odometry as individual odom is not meaningful
        odom_t = self.odoms_t[seq_idx][odom_idx]
        odom = self.odoms[seq_idx][odom_idx]
        # scan_odom_t = self.scan_odoms_t[seq_idx][odom_idx]

        # Flow target
        flow_target = self.flow_targets[seq_idx][odom_idx]

        #Scan transformation
        scan_dir = self.scan_odoms[seq_idx][odom_idx][-1]
        scan_xy = u.rphi_to_xy(scan, scan_phi)
        scan_xy_next = u.rphi_to_xy(scan_next, scan_phi)
        scan_xy = np.stack(scan_xy, axis=1)
        scan_xy_next = np.stack(scan_xy_next, axis=1)

        rot = np.array([[np.cos(odom[-1]), np.sin(odom[-1])],
                        [-np.sin(odom[-1]), np.cos(odom[-1])]], dtype=np.float32)

        rot_trans = np.array([[np.cos(scan_dir), -np.sin(scan_dir)],
                              [np.sin(scan_dir), np.cos(scan_dir)]], dtype=np.float32)

        trans = np.matmul(odom[:-1], rot_trans.T)

        scan_xy_next_rot = np.matmul(scan_xy_next, rot.T) + trans

        # Get dynamic mask
        mask = self._get_dynamic_mask(scan_xy, rtn_dict['dets_wc'], rtn_dict['dets_wa'], rtn_dict['dets_wp']).reshape(-1, 1)

        rtn_dict['scan_pair'] = [scan_xy * mask, scan_xy_next_rot * mask]
        rtn_dict['odom_t'] = odom_t
        rtn_dict['odom'] = odom
        rtn_dict["flow_target"] = flow_target * mask
        # rtn_dict["dynamic_mask"] = np.ones_like(mask)

        return rtn_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            if k in ["scan_pair", "flow_target", "dynamic_mask"]:
                rtn_dict[k] = np.stack([sample[k] for sample in batch], axis=0)
            else:
                rtn_dict[k] = [sample[k] for sample in batch]
        return rtn_dict

    def _load_scan_file(self, seq_name):
        data = np.genfromtxt(seq_name + '.csv', delimiter=",")
        seqs = data[:, 0].astype(np.uint32)
        times = data[:, 1].astype(np.float32)
        scans = data[:, 2:].astype(np.float32)
        return seqs, times, scans

    def _load_odom(self, seq_name):  # --Kevin
        odoms = np.genfromtxt(seq_name + ".odom2", delimiter=",", dtype=[('seq', 'u4'), ('t', 'f4'), ('xya', 'f4', 3)])
        odom_ns = odoms["seq"]
        odom_t = odoms["t"]
        odoms = odoms["xya"]
        return odom_ns, odom_t, odoms

    def _load_odom_file(self, seq_name):
        odoms = np.genfromtxt(seq_name + '.difodom', delimiter=',')
        odoms_t = odoms[:, 0]
        odoms = odoms[:, 1:]

        return odoms_t, odoms

    def _load_flow_file(self, seq_name, shape=(-1, 450, 2)):
        flow_targets = np.genfromtxt(seq_name + '.flow', delimiter=',')

        return flow_targets.reshape(shape)

    def _load_det_file(self, seq_name):
        def do_load(f_name):
            seqs, dets = [], []
            with open(f_name) as f:
                for line in f:
                    seq, tail = line.split(',', 1)
                    seqs.append(int(seq))
                    dets.append(json.loads(tail))
            return seqs, dets

        s1, wcs = do_load(seq_name + '.wc')
        s2, was = do_load(seq_name + '.wa')
        s3, wps = do_load(seq_name + '.wp')
        assert all(a == b == c for a, b, c in zip(s1, s2, s3))

        return np.array(s1), wcs, was, wps

    def _scan_odom_matching(self, seq_idx, scans_t):
        odoms_ns, odoms_t, odoms = [], [], []
        diff_odoms_t, diff_odoms = [], []
        for t in scans_t:
            idx = np.argmin(np.abs(self.odoms_t[seq_idx] - t))
            odom_ns = self.odoms_ns[seq_idx][idx]
            odom_t = self.odoms_t[seq_idx][idx]
            odom = self.odoms[seq_idx][idx]
            diff_odom_t = self.diff_odoms_t[seq_idx][idx]
            diff_odom = self.diff_odoms[seq_idx][idx]
            odoms_ns.append(odom_ns)
            odoms_t.append(odom_t)
            odoms.append(odom)
            diff_odoms_t.append(diff_odom_t)
            diff_odoms.append(diff_odom)

        return odoms_ns, odoms_t, odoms, diff_odoms_t, diff_odoms

    def _get_dynamic_mask(self, scan_xy, dets_wc, dets_wa, dets_wp,
                          radius_wc=0.6, radius_wa=0.5, radius_wp=0.45, n_pts=450):
        all_dets = list(dets_wc) + list(dets_wa) + list(dets_wp)
        all_radius = [radius_wc] * len(dets_wc) + [radius_wa] * len(dets_wa) + [radius_wp] * len(dets_wp)
        mask = np.ones(n_pts, dtype=np.float)

        for det, radius in zip(all_dets, all_radius):
            det_xy = np.hstack(u.rphi_to_xy(det[0], det[1]))
            distance = np.linalg.norm(scan_xy - det_xy, axis=-1)
            mask[distance <= radius] = 0.0
            b = np.argmin(mask)

        a = scan_xy * mask.reshape(-1, 1)

        return mask

# The dataset that remove all static scene
class FlowDatasetTmp2(Dataset):
    def __init__(self, data_path, split='train', testing=False, train_with_val=False):

        if train_with_val:
            seq_names = [f[:-4] for f in glob(os.path.join(data_path, split, '*.csv'))]
            seq_names += [f[:-4] for f in glob(os.path.join(data_path, 'val', '*.csv'))]
        else:
            seq_names = [f[:-4] for f in glob(os.path.join(data_path, split, '*.csv'))]

        seq_names = seq_names[:5]

        if testing:
            # seq_names = seq_names[::3]
            seq_names = seq_names

        flow_targets = [self._load_flow_file(f, shape=(-1, 450, 2)) for f in seq_names]
        non_zero_ids = [np.all(np.all(flow == 0.0, axis=1) == 0.0, axis=1) for flow in flow_targets]

        self.seq_names = []
        flow_targets_reduced = []
        for seq_idx, indices in enumerate(non_zero_ids):
            if np.any(indices):
                self.seq_names.append(seq_names[seq_idx])
                flow_targets_reduced.append(flow_targets[seq_idx])

        non_zero_ids = [np.all(np.all(flow == 0.0, axis=1) == 0.0, axis=1) for flow in flow_targets_reduced]

        if len(self.seq_names) == 0:
            print("No valid data")
            return

        print("{} valid files found".format(len(self.seq_names)))

        # Pre-load scans, annotation, odometry and flow target
        scans_ns, scans_t, scans = zip(*[self._load_scan_file(f) for f in self.seq_names])  # n_seqs * n_scans * n_pts
        scans_next = [np.vstack([s[1:], s[-1].reshape(1, -1)]) for s in scans]
        dets_ns, dets_wc, dets_wa, dets_wp = zip(
            *map(lambda f: self._load_det_file(f), self.seq_names))

        odoms_t, odoms = zip(*[self._load_odom_file(f) for f in self.seq_names])
        _, scan_odoms_t, scan_odoms = zip(*[self._load_odom(f) for f in self.seq_names])

        scans_ns = list(scans_ns)
        scans_t = list(scans_t)
        scans = list(scans)
        dets_ns = list(dets_ns)
        dets_wc = list(dets_wc)
        dets_wa = list(dets_wa)
        dets_wp = list(dets_wp)
        odoms_t = list(odoms_t)
        odoms = list(odoms)
        scan_odoms_t = list(scan_odoms_t)
        scan_odoms = list(scan_odoms)

        self.scans_ns = []
        self.scans_t = []
        self.scans = []
        self.scans_next = []
        self.flow_targets = []

        self.dets_ns = []
        self.dets_wc = []
        self.dets_wa = []
        self.dets_wp = []
        self.odoms_t = []
        self.odoms = []
        self.scan_odoms_t = []
        self.scan_odoms = []

        for seq_idx, indices in enumerate(non_zero_ids):
            if np.any(indices):
                self.scans_ns.append(scans_ns[seq_idx][indices])
                self.scans_t.append(scans_t[seq_idx][indices])
                self.scans.append(scans[seq_idx][indices])
                self.scans_next.append(scans_next[seq_idx][indices])
                self.flow_targets.append(flow_targets_reduced[seq_idx][indices])

                self.dets_ns.append(dets_ns[seq_idx])
                self.dets_wc.append(dets_wc[seq_idx])
                self.dets_wa.append(dets_wa[seq_idx])
                self.dets_wp.append(dets_wp[seq_idx])
                self.odoms_t.append(odoms_t[seq_idx][indices])
                self.odoms.append(odoms[seq_idx][indices])
                self.scan_odoms_t.append(scan_odoms_t[seq_idx][indices])
                self.scan_odoms.append(scan_odoms[seq_idx][indices])

        # Pre-compute mappings from detection index to scan index
        # such that idet2iscan[seq_idx][det_idx] = scan_idx
        self.idet2iscan = []
        self.flat_seq_inds, self.flat_det_inds = [], []
        for seq_idx, ss_ds in enumerate(zip(self.scans_ns, self.dets_ns)):
            ss, ds = ss_ds
            dict = {}
            i = 0
            for d in ds:
                scan_idx = np.where(ss == d)
                if len(scan_idx[0]) > 0:
                    dict[i] = np.where(ss == d)[0][0]
                    i += 1
            self.idet2iscan.append(dict)
            num_samples = len(dict)
            self.flat_seq_inds += [seq_idx] * num_samples
            self.flat_det_inds += range(num_samples)

        # Look-up list for sequence indices and annotation indices.
        # self.flat_seq_inds, self.flat_det_inds = [], []
        # for seq_idx, det_ns in enumerate(self.dets_ns):
        #     num_samples = len(det_ns)
        #     self.flat_seq_inds += [seq_idx] * num_samples
        #     self.flat_det_inds += range(num_samples)

    def __len__(self):
        return len(self.flat_det_inds)

    def __getitem__(self, idx):
        # idx = 10
        seq_idx = self.flat_seq_inds[idx]
        det_idx = self.flat_det_inds[idx]
        dets_ns = self.dets_ns[seq_idx][det_idx]

        rtn_dict = {}
        rtn_dict['seq_name'] = self.seq_names[seq_idx]
        rtn_dict['dets_ns'] = dets_ns

        # Annotation
        rtn_dict['dets_wc'] = self.dets_wc[seq_idx][det_idx]
        rtn_dict['dets_wa'] = self.dets_wa[seq_idx][det_idx]
        rtn_dict['dets_wp'] = self.dets_wp[seq_idx][det_idx]

        # Scan
        scan_idx = self.idet2iscan[seq_idx][det_idx]
        scan = self.scans[seq_idx][scan_idx]
        scan_next = self.scans_next[seq_idx][scan_idx]
        scan_t = self.scans_t[seq_idx][scan_idx]
        odom_idx = np.argmin(np.abs(self.scan_odoms_t[seq_idx] - scan_t))

        # Angle
        scan_phi = u.get_laser_phi()
        rtn_dict['phi_grid'] = scan_phi

        # Odometry. Note that here it refers to difference of odometry as individual odom is not meaningful
        odom_t = self.odoms_t[seq_idx][odom_idx]
        odom = self.odoms[seq_idx][odom_idx]
        # scan_odom_t = self.scan_odoms_t[seq_idx][odom_idx]

        # Flow target
        flow_target = self.flow_targets[seq_idx][odom_idx]

        # Scan transformation
        scan_dir = self.scan_odoms[seq_idx][odom_idx][-1]
        scan_xy = u.rphi_to_xy(scan, scan_phi)
        scan_xy_next = u.rphi_to_xy(scan_next, scan_phi)
        scan_xy = np.stack(scan_xy, axis=1)
        scan_xy_next = np.stack(scan_xy_next, axis=1)

        rot = np.array([[np.cos(odom[-1]), np.sin(odom[-1])],
                        [-np.sin(odom[-1]), np.cos(odom[-1])]], dtype=np.float32)

        rot_trans = np.array([[np.cos(scan_dir), -np.sin(scan_dir)],
                              [np.sin(scan_dir), np.cos(scan_dir)]], dtype=np.float32)

        trans = np.matmul(odom[:-1], rot_trans.T)

        scan_xy_next_rot = np.matmul(scan_xy_next, rot.T) + trans

        # Get dynamic mask
        mask = self._get_dynamic_mask(scan_xy, rtn_dict['dets_wc'], rtn_dict['dets_wa'],
                                      rtn_dict['dets_wp']).reshape(-1, 1)

        rtn_dict['scan_pair'] = [scan_xy * mask, scan_xy_next_rot * mask]
        rtn_dict['odom_t'] = odom_t
        rtn_dict['odom'] = odom
        rtn_dict["flow_target"] = flow_target * mask
        # rtn_dict["dynamic_mask"] = np.ones_like(mask)

        return rtn_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            if k in ["scan_pair", "flow_target", "dynamic_mask"]:
                rtn_dict[k] = np.stack([sample[k] for sample in batch], axis=0)
            else:
                rtn_dict[k] = [sample[k] for sample in batch]
        return rtn_dict

    def _load_scan_file(self, seq_name):
        data = np.genfromtxt(seq_name + '.csv', delimiter=",")
        seqs = data[:, 0].astype(np.uint32)
        times = data[:, 1].astype(np.float32)
        scans = data[:, 2:].astype(np.float32)
        return seqs, times, scans

    def _load_odom(self, seq_name):  # --Kevin
        odoms = np.genfromtxt(seq_name + ".odom2", delimiter=",",
                              dtype=[('seq', 'u4'), ('t', 'f4'), ('xya', 'f4', 3)])
        odom_ns = odoms["seq"]
        odom_t = odoms["t"]
        odoms = odoms["xya"]
        return odom_ns, odom_t, odoms

    def _load_odom_file(self, seq_name):
        odoms = np.genfromtxt(seq_name + '.difodom', delimiter=',')
        odoms_t = odoms[:, 0]
        odoms = odoms[:, 1:]

        return odoms_t, odoms

    def _load_flow_file(self, seq_name, shape=(-1, 450, 2)):
        flow_targets = np.genfromtxt(seq_name + '.flow', delimiter=',')

        return flow_targets.reshape(shape)

    def _load_det_file(self, seq_name):
        def do_load(f_name):
            seqs, dets = [], []
            with open(f_name) as f:
                for line in f:
                    seq, tail = line.split(',', 1)
                    seqs.append(int(seq))
                    dets.append(json.loads(tail))
            return seqs, dets

        s1, wcs = do_load(seq_name + '.wc')
        s2, was = do_load(seq_name + '.wa')
        s3, wps = do_load(seq_name + '.wp')
        assert all(a == b == c for a, b, c in zip(s1, s2, s3))

        return np.array(s1), wcs, was, wps

    def _scan_odom_matching(self, seq_idx, scans_t):
        odoms_ns, odoms_t, odoms = [], [], []
        diff_odoms_t, diff_odoms = [], []
        for t in scans_t:
            idx = np.argmin(np.abs(self.odoms_t[seq_idx] - t))
            odom_ns = self.odoms_ns[seq_idx][idx]
            odom_t = self.odoms_t[seq_idx][idx]
            odom = self.odoms[seq_idx][idx]
            diff_odom_t = self.diff_odoms_t[seq_idx][idx]
            diff_odom = self.diff_odoms[seq_idx][idx]
            odoms_ns.append(odom_ns)
            odoms_t.append(odom_t)
            odoms.append(odom)
            diff_odoms_t.append(diff_odom_t)
            diff_odoms.append(diff_odom)

        return odoms_ns, odoms_t, odoms, diff_odoms_t, diff_odoms

    def _get_dynamic_mask(self, scan_xy, dets_wc, dets_wa, dets_wp,
                          radius_wc=0.6, radius_wa=0.5, radius_wp=0.45, n_pts=450):
        all_dets = list(dets_wc) + list(dets_wa) + list(dets_wp)
        all_radius = [radius_wc] * len(dets_wc) + [radius_wa] * len(dets_wa) + [radius_wp] * len(dets_wp)
        mask = np.ones(n_pts, dtype=np.float)

        for det, radius in zip(all_dets, all_radius):
            det_xy = np.hstack(u.rphi_to_xy(det[0], det[1]))
            distance = np.linalg.norm(scan_xy - det_xy, axis=-1)
            mask[distance <= radius] = 0.0
            b = np.argmin(mask)

        a = scan_xy * mask.reshape(-1, 1)

        return mask

# This dataset remove all static scene. Based on FlowDataset
class FlowDataset2(Dataset):
    def __init__(self, data_path, split='train', testing=False, train_with_val=False):

        if train_with_val:
            seq_names = [f[:-4] for f in glob(os.path.join(data_path, split, '*.csv'))]
            seq_names += [f[:-4] for f in glob(os.path.join(data_path, 'val', '*.csv'))]
        else:
            seq_names = [f[:-4] for f in glob(os.path.join(data_path, split, '*.csv'))]

        seq_names = seq_names[:5]

        if testing:
            # seq_names = seq_names[::5]
            seq_names = seq_names

        flow_targets = [self._load_flow_file(f, shape=(-1, 450, 2)) for f in seq_names]
        a = [np.all(flow == 0.0, axis=1) for flow in flow_targets]
        non_zero_ids = [np.all(np.all(flow == 0.0, axis=1) == 0.0, axis=1) for flow in flow_targets]

        self.seq_names = []
        flow_targets_reduced = []
        for seq_idx, indices in enumerate(non_zero_ids):
            if np.any(indices):
                self.seq_names.append(seq_names[seq_idx])
                flow_targets_reduced.append(flow_targets[seq_idx])

        non_zero_ids = [np.all(np.all(flow == 0.0, axis=1) == 0.0, axis=1) for flow in flow_targets_reduced]

        if len(self.seq_names) == 0:
            print("No valid data")
            return

        print("{} valid files found".format(len(self.seq_names)))

        # Pre-load scans, odometry and flow target
        scans_ns, scans_t, scans = zip(*[self._load_scan_file(f) for f in self.seq_names]) #n_seqs * n_scans * n_pts
        scans_next = [np.vstack([s[1:], s[-1].reshape(1, -1)]) for s in scans]
        # self.scans_ns = np.hstack(self.scans_ns)
        # self.scans_t = np.hstack(self.scans_t)
        # self.scans = np.vstack(self.scans)
        # self.scans_next = np.vstack(self.scans_next)

        odoms_t, odoms = zip(*[self._load_odom_file(f) for f in self.seq_names])
        # self.odoms_t = np.hstack(self.odoms_t)
        # self.odoms = np.vstack(self.odoms)

        _, _, scan_odoms = zip(*[self._load_odom(f) for f in self.seq_names])
        # scan_odoms = np.vstack(scan_odoms)
        # self.scan_dir = scan_odoms[..., -1]

        # n_pts = self.scans.shape[-1]
        # self.flow_targets = [self._load_flow_file(f, shape=(-1, n_pts, 2)) for f in self.seq_names]
        # self.flow_targets = np.vstack(self.flow_targets)

        scans_ns = list(scans_ns)
        scans_t = list(scans_t)
        scans = list(scans)
        odoms_t = list(odoms_t)
        odoms = list(odoms)
        scan_odoms = list(scan_odoms)

        self.scans_ns = []
        self.scans_t = []
        self.scans = []
        self.scans_next = []
        self.flow_targets = []

        self.odoms_t = []
        self.odoms = []
        self.scan_odoms = []

        for seq_idx, indices in enumerate(non_zero_ids):
            if np.any(indices):
                self.scans_ns.append(scans_ns[seq_idx][indices])
                self.scans_t.append(scans_t[seq_idx][indices])
                self.scans.append(scans[seq_idx][indices])
                self.scans_next.append(scans_next[seq_idx][indices])
                self.flow_targets.append(flow_targets_reduced[seq_idx][indices])

                self.odoms_t.append(odoms_t[seq_idx][indices])
                self.odoms.append(odoms[seq_idx][indices])
                self.scan_odoms.append(scan_odoms[seq_idx][indices])

        self.scans_ns = np.hstack(self.scans_ns)
        self.scans_t = np.hstack(self.scans_t)
        self.scans = np.vstack(self.scans)
        self.scans_next = np.vstack(self.scans_next)

        self.odoms_t = np.hstack(self.odoms_t)
        self.odoms = np.vstack(self.odoms)

        self.scan_odoms = np.vstack(self.scan_odoms)
        self.scan_dir = self.scan_odoms[..., -1]

        self.flow_targets = np.vstack(self.flow_targets)

        return

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        rtn_dict = {}
        # rtn_dict['seq_name'] = self.seq_names[idx]

        # Scan
        scan = self.scans[idx]
        scan_next = self.scans_next[idx]

        # Angle
        scan_phi = u.get_laser_phi()
        rtn_dict['phi_grid'] = scan_phi

        # Odometry. Note that here it refers to difference of odometry as individual odom is not meaningful
        odom_t = self.odoms_t[idx]
        odom = self.odoms[idx]

        # Flow target
        flow_target = self.flow_targets[idx]

        #Scan transformation
        scan_dir = self.scan_dir[idx]
        scan_xy = u.rphi_to_xy(scan, scan_phi)
        scan_xy_next = u.rphi_to_xy(scan_next, scan_phi)
        scan_xy = np.stack(scan_xy, axis=1)
        scan_xy_next = np.stack(scan_xy_next, axis=1)

        rot = np.array([[np.cos(odom[-1]), np.sin(odom[-1])],
                        [-np.sin(odom[-1]), np.cos(odom[-1])]], dtype=np.float32)

        rot_trans = np.array([[np.cos(scan_dir), -np.sin(scan_dir)],
                              [np.sin(scan_dir), np.cos(scan_dir)]], dtype=np.float32)

        trans = np.matmul(odom[:-1], rot_trans.T)

        scan_xy_next_rot = np.matmul(scan_xy_next, rot.T) + trans

        rtn_dict['scan_pair'] = [scan_xy, scan_xy_next_rot]
        rtn_dict['odom_t'] = odom_t
        rtn_dict['odom'] = odom
        rtn_dict["flow_target"] = flow_target

        return rtn_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            if k in ["scan_pair", "flow_target"]:
                rtn_dict[k] = np.stack([sample[k] for sample in batch], axis=0)
            else:
                rtn_dict[k] = [sample[k] for sample in batch]

        return rtn_dict

    def _load_scan_file(self, seq_name):
        data = np.genfromtxt(seq_name + '.csv', delimiter=",")
        seqs = data[:, 0].astype(np.uint32)
        times = data[:, 1].astype(np.float32)
        scans = data[:, 2:].astype(np.float32)
        return seqs, times, scans

    def _load_odom(self, seq_name):  # --Kevin
        odoms = np.genfromtxt(seq_name + ".odom2", delimiter=",", dtype=[('seq', 'u4'), ('t', 'f4'), ('xya', 'f4', 3)])
        odom_ns = odoms["seq"]
        odom_t = odoms["t"]
        odoms = odoms["xya"]
        return odom_ns, odom_t, odoms

    def _load_odom_file(self, seq_name):
        odoms = np.genfromtxt(seq_name + '.difodom', delimiter=',')
        odoms_t = odoms[:, 0]
        odoms = odoms[:, 1:]

        return odoms_t, odoms

    def _load_flow_file(self, seq_name, shape=(-1, 450, 2)):
        flow_targets = np.genfromtxt(seq_name + '.flow', delimiter=',')

        return flow_targets.reshape(shape)

    def _scan_odom_matching(self, seq_idx, scans_t):
        odoms_ns, odoms_t, odoms = [], [], []
        diff_odoms_t, diff_odoms = [], []
        for t in scans_t:
            idx = np.argmin(np.abs(self.odoms_t[seq_idx] - t))
            odom_ns = self.odoms_ns[seq_idx][idx]
            odom_t = self.odoms_t[seq_idx][idx]
            odom = self.odoms[seq_idx][idx]
            diff_odom_t = self.diff_odoms_t[seq_idx][idx]
            diff_odom = self.diff_odoms[seq_idx][idx]
            odoms_ns.append(odom_ns)
            odoms_t.append(odom_t)
            odoms.append(odom)
            diff_odoms_t.append(diff_odom_t)
            diff_odoms.append(diff_odom)

        return odoms_ns, odoms_t, odoms, diff_odoms_t, diff_odoms

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = FlowDataset(data_path='../../../data/DROWv2-data')

    sample = iter(dataset)
    rtn_dict = next(sample)

    import cv2

    data_path = "./data/DROWv2-data"
    split = "train"

    tmp_imgs = os.path.join("/", "tmp_imgs")
    os.makedirs(tmp_imgs, exist_ok=True)
    tmp_videos = os.path.join("/", "tmp_videos")
    os.makedirs(tmp_videos, exist_ok=True)

    scans = rtn_dict["scan"]
    scan_phi = rtn_dict["phi_grid"]
    flow_targets = rtn_dict["flow_target"]

    scans_sample = scans[:500]
    flow_sample = flow_targets[:500]

    scan_path = './../../tmp_videos/scan_samples.avi'
    vu.plot_sequence(scans_sample, np.repeat(scan_phi.reshape(1, -1), len(scans_sample), axis=0), path=scan_path)

    flow_arrow_path = './../../tmp_videos/flow_arrow_samples.avi'
    vu.plot_sequence(scans_sample, np.repeat(scan_phi.reshape(1, -1), len(scans_sample), axis=0), path=flow_arrow_path, arrow=flow_sample)

    flow_hsv = "./../../tmp_videos/flow_hsv_samples.avi"
    flow_hsv_opencv = "./../../tmp_videos/flow_hsv_samples_opencv.avi"
    scans_sample,
    colors = []
    for flow in flow_sample:
        hsv = u.flow_to_hsv(flow)
        colors.append(hsv)
    colors = np.asarray(colors)
    vu.plot_sequence(scans_sample, np.repeat(scan_phi.reshape(1, -1), len(scans_sample), axis=0), path=flow_hsv, color=colors)
