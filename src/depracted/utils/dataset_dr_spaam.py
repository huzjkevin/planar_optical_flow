from glob import glob
import os

import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

from . import utils as u
import matplotlib.pyplot as plt


def create_dataloader(data_path, num_scans, batch_size, num_workers, network_type="cutout",
                      train_with_val=False, use_data_augumentation=False,
                      cutout_kwargs=None, polar_grid_kwargs=None,
                      pedestrian_only=False):
    train_set = DROWDataset2(data_path=data_path,
                            split='train',
                            num_scans=num_scans,
                            network_type=network_type,
                            train_with_val=train_with_val,
                            use_data_augumentation=use_data_augumentation,
                            cutout_kwargs=cutout_kwargs,
                            polar_grid_kwargs=polar_grid_kwargs,
                            pedestrian_only=pedestrian_only)

    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True,
                              num_workers=num_workers, shuffle=True,
                              collate_fn=train_set.collate_batch)
    if train_with_val:
        eval_set = DROWDataset2(data_path=data_path,
                               split='val',
                               num_scans=num_scans,
                               network_type=network_type,
                               train_with_val=False,
                               use_data_augumentation=False,
                               cutout_kwargs=cutout_kwargs,
                               polar_grid_kwargs=polar_grid_kwargs,
                               pedestrian_only=pedestrian_only)
        
        eval_loader = DataLoader(eval_set, batch_size=batch_size, pin_memory=True,
                                num_workers=num_workers, shuffle=True,
                                collate_fn=eval_set.collate_batch)
        return train_loader, eval_loader
    else:
        return train_loader, None


def create_test_dataloader(data_path, num_scans, network_type="cutout",
                           cutout_kwargs=None, polar_grid_kwargs=None,
                           pedestrian_only=False, split='test',
                           scan_stride=1, pt_stride=1):

    test_set = DROWDataset2(data_path=data_path,
                           split=split,
                           num_scans=num_scans,
                           network_type=network_type,
                           train_with_val=False,
                           use_data_augumentation=False,
                           cutout_kwargs=cutout_kwargs,
                           polar_grid_kwargs=polar_grid_kwargs,
                           pedestrian_only=pedestrian_only,
                           scan_stride=scan_stride,
                           pt_stride=pt_stride)

    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True,
                             num_workers=1, shuffle=False,
                             collate_fn=test_set.collate_batch)
    return test_loader


# class DROWDataset(Dataset):
#     def __init__(self, data_path, split='train', num_scans=5, network_type="cutout",
#                  train_with_val=False, cutout_kwargs=None, polar_grid_kwargs=None,
#                  use_data_augumentation=False, pedestrian_only=False,
#                  scan_stride=1, pt_stride=1):
#         self._num_scans = num_scans
#         self._use_data_augmentation = use_data_augumentation
#         self._cutout_kwargs = cutout_kwargs
#         self._network_type = network_type
#         self._polar_grid_kwargs = polar_grid_kwargs
#         self._pedestrian_only = pedestrian_only
#         self._scan_stride = scan_stride
#         self._pt_stride = pt_stride  # @TODO remove pt_stride
#
#         if train_with_val:
#             seq_names = [f[:-4] for f in glob(os.path.join(data_path, 'train', '*.csv'))]
#             seq_names += [f[:-4] for f in glob(os.path.join(data_path, 'val', '*.csv'))]
#         else:
#             seq_names = [f[:-4] for f in glob(os.path.join(data_path, split, '*.csv'))]
#
#         seq_names = seq_names[::5]
#         self.seq_names = seq_names
#
#         # Pre-load scans odoms, and annotations
#         self.scans_ns, self.scans_t, self.scans = zip(*[self._load_scan_file(f) for f in seq_names])
#         self.dets_ns, self.dets_wc, self.dets_wa, self.dets_wp = zip(*map(
#             lambda f: self._load_det_file(f), seq_names))
#         self.odoms_t, self.odoms = zip(*[self._load_odom_file(f) for f in seq_names])
#         _, self.scan_odoms_t, self.scan_odoms = zip(*[self._load_odom(f) for f in self.seq_names])
#
#         # Pre-compute mappings from detection index to scan index
#         # such that idet2iscan[seq_idx][det_idx] = scan_idx
#         self.idet2iscan = [{i: np.where(ss == d)[0][0] for i, d in enumerate(ds)}
#                 for ss, ds in zip(self.scans_ns, self.dets_ns)]
#
#         # Look-up list for sequence indices and annotation indices.
#         self.flat_seq_inds, self.flat_det_inds = [], []
#         for seq_idx, det_ns in enumerate(self.dets_ns):
#             num_samples = len(det_ns)
#             self.flat_seq_inds += [seq_idx] * num_samples
#             self.flat_det_inds += range(num_samples)
#
#     def __len__(self):
#         return len(self.flat_det_inds)
#
#     def __getitem__(self, idx):
#         # idx = 10
#         seq_idx = self.flat_seq_inds[idx]
#         det_idx = self.flat_det_inds[idx]
#         dets_ns = self.dets_ns[seq_idx][det_idx]
#
#         rtn_dict = {}
#         rtn_dict['seq_name'] = self.seq_names[seq_idx]
#         rtn_dict['dets_ns'] = dets_ns
#
#         # Annotation
#         rtn_dict['dets_wc'] = self.dets_wc[seq_idx][det_idx]
#         rtn_dict['dets_wa'] = self.dets_wa[seq_idx][det_idx]
#         rtn_dict['dets_wp'] = self.dets_wp[seq_idx][det_idx]
#
#         # Scan
#         scan_idx = self.idet2iscan[seq_idx][det_idx]
#         scan_next_idx = min(scan_idx + 1, len(self.scans[seq_idx]) - 1)
#         inds_tmp = (np.arange(self._num_scans) * self._scan_stride)[::-1]
#         scan_inds = [max(0, scan_idx - i) for i in inds_tmp]
#         scans = np.array([self.scans[seq_idx][i] for i in scan_inds])
#         scans = scans[:, ::self._pt_stride]
#         scans_ns = [self.scans_ns[seq_idx][i] for i in scan_inds]
#         scan_next = self.scans[seq_idx][scan_next_idx]
#         scan_next_ns = self.scans_ns[seq_idx][scan_next_idx]
#
#         # rtn_dict['scans'] = np.vstack([scans, scan_next])
#         # rtn_dict['scans_ns'] = np.hstack([scans_ns, scan_next_ns])
#
#         rtn_dict['scans'] = scans
#         rtn_dict['scans_ns'] = scans_ns
#
#         # Odom
#         scan_t = self.scans_t[seq_idx][scan_idx]
#         odom_idx = np.argmin(np.abs(self.scan_odoms_t[seq_idx] - scan_t))
#         odom_t = self.odoms_t[seq_idx][odom_idx]
#         odom = self.odoms[seq_idx][odom_idx]
#         rtn_dict['odom_t'] = odom_t
#         rtn_dict['odom'] = odom
#
#         # angle
#         scan_phi = u.get_laser_phi()[::self._pt_stride]
#         rtn_dict['phi_grid'] = scan_phi
#
#         # Regression target
#         target_cls, target_reg = u.get_regression_target(
#                 scans[-1],
#                 scan_phi,
#                 rtn_dict['dets_wc'],
#                 rtn_dict['dets_wa'],
#                 rtn_dict['dets_wp'],
#                 pedestrian_only=self._pedestrian_only)
#
#         rtn_dict['target_cls'] = target_cls
#         rtn_dict['target_reg'] = target_reg
#
#         # Flow target
#         target_flow = u.get_flow_target_canonical(scans[-1], scan_phi, odom_t, odom)
#         rtn_dict['target_flow'] = target_flow
#
#         if self._use_data_augmentation:
#             rtn_dict = u.data_augmentation(rtn_dict)
#
#         # polar grid or cutout
#         if self._network_type == "cutout" \
#                 or self._network_type == "cutout_gating" \
#                 or self._network_type == "cutout_spatial":
#             if "area_mode" not in self._cutout_kwargs:
#                 cutout = u.scans_to_cutout_original(
#                     rtn_dict['scans'], scan_phi[1] - scan_phi[0],
#                     **self._cutout_kwargs)
#             else:
#                 cutout = u.scans_to_cutout(rtn_dict['scans'], scan_phi, stride=1,
#                                            **self._cutout_kwargs)
#             rtn_dict['input'] = cutout
#         elif self._network_type == "fc1d":
#             rtn_dict['input'] = np.expand_dims(scans, axis=1)
#         elif self._network_type == 'fc1d_fea':
#             cutout = u.scans_to_cutout(rtn_dict['scans'],
#                                        scan_phi[1] - scan_phi[0],
#                                        **self._cutout_kwargs)
#             rtn_dict['input'] = np.transpose(cutout, (1, 2, 0))
#         elif self._network_type == "fc2d":
#             polar_grid = u.scans_to_polar_grid(rtn_dict['scans'],
#                                                **self._polar_grid_kwargs)
#             rtn_dict['input'] = np.expand_dims(polar_grid, axis=1)
#         elif self._network_type == 'fc2d_fea':
#             raise NotImplementedError
#
#         return rtn_dict
#
#     def collate_batch(self, batch):
#         rtn_dict = {}
#         for k, _ in batch[0].items():
#             if k in ['scans', 'target_cls', 'target_reg', 'input', 'target_flow']:
#                 rtn_dict[k] = np.array([sample[k] for sample in batch])
#             else:
#                 rtn_dict[k] = [sample[k] for sample in batch]
#
#         return rtn_dict
#
#     def _load_scan_file(self, seq_name):
#         data = np.genfromtxt(seq_name + '.csv', delimiter=",")
#         seqs = data[:, 0].astype(np.uint32)
#         times = data[:, 1].astype(np.float32)
#         scans = data[:, 2:].astype(np.float32)
#         return seqs, times, scans
#
#     def _load_det_file(self, seq_name):
#         def do_load(f_name):
#             seqs, dets = [], []
#             with open(f_name) as f:
#                 for line in f:
#                     seq, tail = line.split(',', 1)
#                     seqs.append(int(seq))
#                     dets.append(json.loads(tail))
#             return seqs, dets
#
#         s1, wcs = do_load(seq_name + '.wc')
#         s2, was = do_load(seq_name + '.wa')
#         s3, wps = do_load(seq_name + '.wp')
#         assert all(a == b == c for a, b, c in zip(s1, s2, s3))
#
#         return np.array(s1), wcs, was, wps
#
#     def _load_odom_file(self, seq_name):
#         odoms = np.genfromtxt(seq_name + '.difodom', delimiter=',')
#         odoms_t = odoms[:, 0]
#         odoms = odoms[:, 1:]
#
#         return odoms_t, odoms
#
#     def _load_odom(self, seq_name):
#         odoms = np.genfromtxt(seq_name + ".odom2", delimiter=",", dtype=[('seq', 'u4'), ('t', 'f4'), ('xya', 'f4', 3)])
#         odom_ns = odoms["seq"]
#         odom_t = odoms["t"]
#         odoms = odoms["xya"]
#         return odom_ns, odom_t, odoms

# This dataset removes all static sequences
class DROWDataset2(Dataset):
    def __init__(self, data_path, split='train', num_scans=5, network_type="cutout",
                 train_with_val=False, cutout_kwargs=None, polar_grid_kwargs=None,
                 use_data_augumentation=False, pedestrian_only=False,
                 scan_stride=1, pt_stride=1, max_scan_dist=6):
        self._num_scans = num_scans
        self._use_data_augmentation = use_data_augumentation
        self._cutout_kwargs = cutout_kwargs
        self._network_type = network_type
        self._polar_grid_kwargs = polar_grid_kwargs
        self._pedestrian_only = pedestrian_only
        self._scan_stride = scan_stride
        self._pt_stride = pt_stride  # @TODO remove pt_stride
        self.max_scan_dist = max_scan_dist # the max distance between the indices of the two scans used for computing target flow

        seq_names = [f[:-4] for f in glob(os.path.join(data_path, split, '*.csv'))]

        seq_names = seq_names[:5]

        _, odoms_t, odoms = zip(*[self._load_odom(f) for f in seq_names])

        #Get rid of static scene
        non_zero_ids = []
        self.seq_names = []
        self.odoms_t, self.odoms = [], []
        for idx, (odom_t, odom) in enumerate(zip(odoms_t, odoms)):

            indices = np.hstack([np.any((odom[1:] - odom[:-1]) != 0.0, axis=1), False])
            # indices = np.hstack([np.abs((odom[1:] - odom[:-1]))[:, -1] >= 0.01, False])
            if not np.any(indices):
                continue
            non_zero_ids.append(indices)
            self.seq_names.append(seq_names[idx])
            self.odoms_t.append(odom_t[indices])
            self.odoms.append(odom[indices])

        if len(self.seq_names) == 0:
            raise FileNotFoundError('{}: No valid data'.format(split))

        print("{}: {} valid files found".format(split, len(self.seq_names)))

        # Pre-load scans odoms, and annotations
        scans_ns, scans_t, scans = zip(*[self._load_scan_file(f) for f in self.seq_names])
        dets_ns, dets_wc, dets_wa, dets_wp = zip(*map(lambda f: self._load_det_file(f), self.seq_names))

        self.scans_ns = []
        self.scans_t = []
        self.scans = []

        self.dets_ns = []
        self.dets_wc = []
        self.dets_wa = []
        self.dets_wp = []

        for seq_idx, indices in enumerate(non_zero_ids):
            if np.any(indices):
                self.scans_ns.append(scans_ns[seq_idx][indices])
                self.scans_t.append(scans_t[seq_idx][indices])
                self.scans.append(scans[seq_idx][indices])
                self.dets_ns.append(dets_ns[seq_idx])
                self.dets_wc.append(dets_wc[seq_idx])
                self.dets_wa.append(dets_wa[seq_idx])
                self.dets_wp.append(dets_wp[seq_idx])

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

    def __len__(self):
        return len(self.flat_det_inds)

    def __getitem__(self, idx):
        # idx = 25
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

        # Scan. Look back 10 + distance scans. Use the first 10 scans for template.
        # Use current scan and the 10th scan for target flow and generating masks.
        # distance = np.random.randint(1, self.max_scan_dist)
        distance = 5
        scan_idx = self.idet2iscan[seq_idx][det_idx]
        cur_scan = self.scans[seq_idx][scan_idx]
        inds_tmp = (np.arange(self._num_scans + distance) * self._scan_stride)[::-1]
        scan_inds = [max(0, scan_idx - i) for i in inds_tmp[:self._num_scans]]
        scans = np.array([self.scans[seq_idx][i] for i in scan_inds])
        scans = scans[:, ::self._pt_stride]
        scans_ns = [self.scans_ns[seq_idx][i] for i in scan_inds]

        rtn_dict['scans'] = np.vstack((scans, cur_scan))
        rtn_dict['scans_ns'] = scans_ns

        # Odom
        scan1_t = self.scans_t[seq_idx][scan_idx]
        scan0_t = self.scans_t[seq_idx][scan_inds[-1]]
        odom1_idx = np.argmin(np.abs(self.odoms_t[seq_idx] - scan1_t))
        odom0_idx = np.argmin(np.abs(self.odoms_t[seq_idx] - scan0_t))

        odom1_t = self.odoms_t[seq_idx][odom1_idx]
        odom0_t = self.odoms_t[seq_idx][odom0_idx]
        odom1 = self.odoms[seq_idx][odom1_idx]
        odom0 = self.odoms[seq_idx][odom0_idx]

        rtn_dict['odom1_t'] = odom1_t
        rtn_dict['odom1'] = odom1

        # angle
        scan_phi = u.get_laser_phi()[::self._pt_stride]
        rtn_dict['phi_grid'] = scan_phi

        # Regression target
        target_cls, target_reg = u.get_regression_target(
            cur_scan,
            scan_phi,
            rtn_dict['dets_wc'],
            rtn_dict['dets_wa'],
            rtn_dict['dets_wp'],
            pedestrian_only=self._pedestrian_only)

        rtn_dict['target_cls'] = target_cls
        rtn_dict['target_reg'] = target_reg

        # Flow target
        cur_scan_xy = np.array(u.rphi_to_xy(cur_scan, scan_phi)).T
        target_flow = u.get_displacement_from_odometry(cur_scan_xy, odom0, odom1)
        target_flow = u.global_to_canonical_flow(target_flow, scan_phi)
        rtn_dict['target_flow'] = target_flow

        # Get dynamic mask
        dynamic_mask = self._get_dynamic_mask(cur_scan_xy, rtn_dict['dets_wc'], rtn_dict['dets_wa'], rtn_dict['dets_wp'])
        valid_point_mask = self._get_valid_point_mask(cur_scan)
        exclude_mask = dynamic_mask * valid_point_mask
        rtn_dict['exclude_mask'] = exclude_mask

        # #=========test============
        # fig = plt.figure(figsize=(9, 6))
        # ax = fig.add_subplot(111)
        #
        # plt.cla()
        # ax.set_aspect('equal')
        # ax.set_xlim(-35, 35)
        # ax.set_ylim(-35, 35)
        #
        # dynamic = scan_xy[dynamic_mask == 0.0]
        # valid = scan_xy[dynamic_mask * valid_point_mask == 1.0]
        # invalid = scan_xy[valid_point_mask == 0.0]
        #
        # ax.scatter(valid[..., 0], valid[..., 1], s=3, c='black')
        # ax.scatter(dynamic[..., 0], dynamic[..., 1], s=3, c='red')
        # ax.scatter(invalid[..., 0], invalid[..., 1], s=3, c='blue')
        # # ax.quiver(scan_xy[..., 0], scan_xy[..., 1], target_flow[..., 0], target_flow[..., 1])
        # plt.savefig('./tmp_imgs/frame_%04d.png' % idx)
        # plt.show()
        #
        # #========================

        if self._use_data_augmentation:
            rtn_dict = u.data_augmentation(rtn_dict)

        # polar grid or cutout
        if self._network_type == "cutout" \
                or self._network_type == "cutout_gating" \
                or self._network_type == "cutout_spatial":
            if "area_mode" not in self._cutout_kwargs:
                cutout = u.scans_to_cutout_original(
                    rtn_dict['scans'], scan_phi[1] - scan_phi[0],
                    **self._cutout_kwargs)
            else:
                cutout = u.scans_to_cutout(rtn_dict['scans'], scan_phi, stride=1,
                                           **self._cutout_kwargs)
            rtn_dict['input'] = cutout
        elif self._network_type == "fc1d":
            rtn_dict['input'] = np.expand_dims(scans, axis=1)
        elif self._network_type == 'fc1d_fea':
            cutout = u.scans_to_cutout(rtn_dict['scans'],
                                       scan_phi[1] - scan_phi[0],
                                       **self._cutout_kwargs)
            rtn_dict['input'] = np.transpose(cutout, (1, 2, 0))
        elif self._network_type == "fc2d":
            polar_grid = u.scans_to_polar_grid(rtn_dict['scans'],
                                               **self._polar_grid_kwargs)
            rtn_dict['input'] = np.expand_dims(polar_grid, axis=1)
        elif self._network_type == 'fc2d_fea':
            raise NotImplementedError

        return rtn_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            if k in ['scans', 'target_cls', 'target_reg', 'input', 'target_flow', 'exclude_mask', 'odom']:
                rtn_dict[k] = np.array([sample[k] for sample in batch])
            else:
                rtn_dict[k] = [sample[k] for sample in batch]
        return rtn_dict

    def _load_scan_file(self, seq_name):
        data = np.genfromtxt(seq_name + '.csv', delimiter=",")
        seqs = data[:, 0].astype(np.uint32)
        times = data[:, 1].astype(np.float32)
        scans = data[:, 2:].astype(np.float32)
        return seqs, times, scans

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

    def _load_odom_file(self, seq_name):
        odoms = np.genfromtxt(seq_name + '.difodom', delimiter=',')
        odoms_t = odoms[:, 0]
        odoms = odoms[:, 1:]

        return odoms_t, odoms

    def _load_odom(self, seq_name):
        odoms = np.genfromtxt(seq_name + ".odom2", delimiter=",", dtype=[('seq', 'u4'), ('t', 'f4'), ('xya', 'f4', 3)])
        odom_ns = odoms["seq"]
        odom_t = odoms["t"]
        odoms = odoms["xya"]
        return odom_ns, odom_t, odoms

    def _get_dynamic_mask(self, scan_xy, dets_wc, dets_wa, dets_wp,
                          radius_wc=2.5, radius_wa=2.0, radius_wp=2.0, n_pts=450):

        all_dets = list(dets_wc) + list(dets_wa) + list(dets_wp)
        all_radius = [radius_wc] * len(dets_wc) + [radius_wa] * len(dets_wa) + [radius_wp] * len(dets_wp)
        mask = np.ones(n_pts, dtype=np.float)

        for det, radius in zip(all_dets, all_radius):
            det_xy = np.hstack(u.rphi_to_xy(det[0], det[1]))
            distance = np.linalg.norm(scan_xy - det_xy, axis=-1)
            mask[distance <= radius] = 0.0

        return mask

    def _get_valid_point_mask(self, scan, thresh=20.0):
        mask = np.ones_like(scan)
        mask[scan >= 20.0] = 0.0

        return mask

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = DROWDataset2(data_path='../data/DROWv2-data')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for sample in dataset:
        target_cls, target_reg = sample['target_cls'], sample['target_reg']
        scans = sample['scans']
        scan_phi = u.get_laser_phi()

        num_scans = scans.shape[0]
        for scan_idx in range(1):
            scan_x, scan_y = u.scan_to_xy(scans[-scan_idx])

            plt.cla()
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.scatter(scan_x, scan_y, s=1, c='black')

            colors = ['blue', 'green', 'red']
            cls_labels = [1, 2, 3]
            for cls_label, c in zip(cls_labels, colors):
                canonical_dxy = target_reg[target_cls==cls_label]
                dets_r, dets_phi = u.canonical_to_global(
                        scans[-1][target_cls==cls_label],
                        scan_phi[target_cls==cls_label],
                        canonical_dxy[:, 0],
                        canonical_dxy[:, 1])
                dets_x, dets_y = u.rphi_to_xy(dets_r, dets_phi)
                ax.scatter(dets_x, dets_y, s=5, c=c)

            plt.pause(0.1)

    plt.show()
