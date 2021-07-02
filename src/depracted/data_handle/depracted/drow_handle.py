from glob import glob
import json
import numpy as np
import os


class DROWHandle:
    def __init__(self, split, cfg):
        self._num_scans = cfg["num_scans"]
        self._scan_stride = cfg["scan_stride"]
        data_dir = os.path.abspath(os.path.expanduser(cfg["data_dir"]))

        seq_names = [f[:-4] for f in glob(os.path.join(data_dir, split, "*.csv"))]
        seq_names = seq_names[:3]
        # seq_names = ['./data/DROWv2-data/test/run_t_2015-11-26-11-22-03.bag']
        self.seq_names = seq_names

        # preload all annotations
        self.dets_ns, self.dets_wc, self.dets_wa, self.dets_wp = zip(
            *map(lambda f: self._load_det_file(f), seq_names))

        _, self.odoms_t, self.odoms = zip(*[self._load_odom(f) for f in seq_names])

        # look-up list to convert an index to sequence index and annotation index
        self.__flat_seq_inds, self.__flat_det_inds = [], []
        for seq_idx, det_ns in enumerate(self.dets_ns):
            num_samples = len(det_ns)
            self.__flat_seq_inds += [seq_idx] * num_samples
            self.__flat_det_inds += range(num_samples)

        # placeholder for scans, which will be preload on the fly
        self.scans_ns = [None] * len(seq_names)
        self.scans_t = [None] * len(seq_names)
        self.scans = [None] * len(seq_names)

        # placeholder for mapping from detection index to scan index
        self.__id2is = [None] * len(seq_names)

        # load the scan sequence into memory if it has not been loaded
        for seq_idx in range(len(self.seq_names)):
            if self.scans[seq_idx] is None:
                self._load_scan_sequence(seq_idx)

    def __len__(self):
        return len(self.__flat_det_inds)

    def __getitem__(self, idx):
        # find matching seq_idx, det_idx, and scan_idx
        seq_idx = self.__flat_seq_inds[idx]
        det_idx = self.__flat_det_inds[idx]
        scan_idx = self.__id2is[seq_idx][det_idx]

        rtn_dict = {}
        rtn_dict["seq_name"] = self.seq_names[seq_idx]
        rtn_dict["det_idx"] = det_idx
        rtn_dict["scan_idx"] = scan_idx

        # annotation [(r, phi), (r, phi), ...]
        rtn_dict["dets_wc"] = self.dets_wc[seq_idx][det_idx]
        rtn_dict["dets_wa"] = self.dets_wa[seq_idx][det_idx]
        rtn_dict["dets_wp"] = self.dets_wp[seq_idx][det_idx]

        # load sequential scans up to the current one (array[frame, point])
        delta_inds = (np.arange(self._num_scans) * self._scan_stride)[::-1]
        scans_inds = [max(0, scan_idx - i) for i in delta_inds]
        rtn_dict["scans"] = np.array([self.scans[seq_idx][i] for i in scans_inds])
        rtn_dict["scans_ind"] = scans_inds
        rtn_dict["scan_phi"] = self.get_laser_phi()

        # load odom
        scan_t = self.scans_t[seq_idx][scan_idx]
        odom_idx = np.argmin(np.abs(self.odoms_t[seq_idx] - scan_t))
        rtn_dict['odom_t'] = self.odoms_t[seq_idx][odom_idx]

        return rtn_dict

    def _load_scan_sequence(self, seq_idx):
        data = np.genfromtxt(self.seq_names[seq_idx] + ".csv", delimiter=",")
        self.scans_ns[seq_idx] = data[:, 0].astype(np.uint32)
        self.scans_t[seq_idx] = data[:, 1].astype(np.float32)
        self.scans[seq_idx] = data[:, 2:].astype(np.float32)

        # precompute a mapping from detection index to scan index such that
        # scans[seq_idx][scan_idx] matches dets[seq_idx][det_idx]
        is_ = 0
        id2is = []
        for det_ns in self.dets_ns[seq_idx]:
            while self.scans_ns[seq_idx][is_] != det_ns:
                is_ += 1
            id2is.append(is_)
        self.__id2is[seq_idx] = id2is

    def _load_det_file(self, seq_name):
        def do_load(f_name):
            seqs, dets = [], []
            with open(f_name) as f:
                for line in f:
                    seq, tail = line.split(",", 1)
                    seqs.append(int(seq))
                    dets.append(json.loads(tail))
            return seqs, dets

        s1, wcs = do_load(seq_name + ".wc")
        s2, was = do_load(seq_name + ".wa")
        s3, wps = do_load(seq_name + ".wp")
        assert all(a == b == c for a, b, c in zip(s1, s2, s3))

        return np.array(s1), wcs, was, wps

    def _load_odom(self, seq_name):
        odoms = np.genfromtxt(seq_name + ".odom2", delimiter=",", dtype=[('seq', 'u4'), ('t', 'f4'), ('xya', 'f4', 3)])
        odom_ns = odoms["seq"]
        odom_t = odoms["t"]
        odoms = odoms["xya"]
        return odom_ns, odom_t, odoms

    @staticmethod
    def get_laser_phi(angle_inc=np.radians(0.5), num_pts=450):
        # Default setting of DROW, which use SICK S300 laser, with 225 deg fov
        # and 450 pts, mounted at 37cm height.
        laser_fov = (num_pts - 1) * angle_inc  # 450 points
        return np.linspace(-laser_fov * 0.5, laser_fov * 0.5, num_pts)
