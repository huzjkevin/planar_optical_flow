# from glob import glob
import json
import numpy as np
import os
# from ._pypcd import point_cloud_from_path
from src.depracted.data_handle._pypcd import point_cloud_from_path
from src.utils.utils import rphi_to_xy
import src.utils.jrdb_transforms as jt

# NOTE: Don't use open3d to load point cloud since it spams the console. Setting
# verbosity level does not solve the problem
# https://github.com/intel-isl/Open3D/issues/1921
# https://github.com/intel-isl/Open3D/issues/884

# Customized train-val split
_JRDB_TRAIN_SEQUENCES = [
    "packard-poster-session-2019-03-20_2",
    "packard-poster-session-2019-03-20_1",
    "clark-center-intersection-2019-02-28_0",
    "huang-lane-2019-02-12_0",
    "jordan-hall-2019-04-22_0",
    "memorial-court-2019-03-16_0",
    "packard-poster-session-2019-03-20_0",
    "clark-center-2019-02-28_1",
    "stlc-111-2019-04-19_0",
    "clark-center-2019-02-28_0",
    "tressider-2019-03-16_0",
    "svl-meeting-gates-2-2019-04-08_1",
    "forbes-cafe-2019-01-22_0",
    "gates-159-group-meeting-2019-04-03_0",
    "huang-basement-2019-01-25_0",
    "svl-meeting-gates-2-2019-04-08_0",
    "tressider-2019-03-16_1",
    "nvidia-aud-2019-04-18_0",
]

_JRDB_VAL_SEQUENCES = [
    "cubberly-auditorium-2019-04-22_0",
    "tressider-2019-04-26_2",
    "gates-to-clark-2019-02-28_1",
    "meyer-green-2019-03-16_0",
    "gates-basement-elevators-2019-01-17_1",
    "huang-2-2019-01-25_0",
    "bytes-cafe-2019-02-07_0",
    "hewlett-packard-intersection-2019-01-24_0",
    "gates-ai-lab-2019-02-08_0",
]



class JRDBHandle:
    def __init__(self, split, cfg):
        assert split in ["train", "val", "test"], f'Invalid split "{split}"'
        self.__num_scan = cfg["num_scan"]
        self.__scan_stride = cfg["scan_stride"]

        data_dir = os.path.abspath(os.path.expanduser(cfg["data_dir"]))

        if split in ['train', 'val']:
            data_dir = os.path.join(data_dir, 'train' + "_dataset")
        else:
            data_dir = os.path.join(data_dir, split + "_dataset")

        self.data_dir = data_dir
        self.timestamp_dir = os.path.join(data_dir, "timestamps")
        self.pc_label_dir = os.path.join(data_dir, "labels", "labels_3d")
        self.im_label_dir = os.path.join(data_dir, "labels", "labels_2d_stitched")

        self.sequence_names = (_JRDB_TRAIN_SEQUENCES if split == "train" else _JRDB_VAL_SEQUENCES)

        self.sequence_names = os.listdir(self.timestamp_dir)
        self.sequence_pc_frames = []
        self.sequence_im_frames = []
        self.sequence_pc_labels = []
        self.sequence_im_labels = []
        self.__flat_inds_sequence = []
        self.__flat_inds_frame = []
        for idx, seq_name in enumerate(self.sequence_names):
            pc_frames, im_frames, pc_labels, im_labels = self._load_one_sequence(seq_name)
            self.sequence_pc_frames.append(pc_frames)
            self.sequence_im_frames.append(im_frames)
            self.sequence_pc_labels.append(pc_labels)
            self.sequence_im_labels.append(im_labels)

            # only use frames that has annotation
            pc_labeled_frames = []
            for fr_idx, fr in enumerate(pc_frames):
                fname = os.path.basename(fr["pointclouds"]["upper_velodyne"]["url"])
                if fname in pc_labels:
                    pc_labeled_frames.append(fr_idx)

            # build a flat index for all sequences and frames
            sequence_length = len(pc_labeled_frames)
            self.__flat_inds_sequence += sequence_length * [idx]
            self.__flat_inds_frame += pc_labeled_frames

    def __len__(self):
        return len(self.__flat_inds_frame)

    def __getitem__(self, idx):
        idx_sq = self.__flat_inds_sequence[idx]
        idx_fr = self.__flat_inds_frame[idx]

        pc_frame = self.sequence_pc_frames[idx_sq][idx_fr]
        pc_data = {}
        for k, v in pc_frame["pointclouds"].items():
            pc_data[k] = self._load_pointcloud(v["url"])

        laser_data = self._load_consecutive_lasers(pc_frame["laser"]["url"])

        pc_fname = os.path.basename(pc_frame["pointclouds"]["upper_velodyne"]["url"])
        anns = self.sequence_pc_labels[idx_sq][pc_fname]

        pc_frame.update(
            {
                "pc_data": pc_data,
                "laser_data": laser_data,
                "anns": anns,
                "laser_grid": np.linspace(
                    -np.pi, np.pi, laser_data.shape[1], dtype=np.float32
                ),
                "laser_z": -0.7 * np.ones(laser_data.shape[1], dtype=np.float32),
            }
        )

        return pc_frame

    def _load_one_sequence(self, seq_name):
        """Load frame info and labels for one sequence.

        Args:
            seq_name (str): Sequence name

        Returns:
            pc_frames (list[dict]): Each element in the list contains information
                for the pointcloud at a single frame
            im_frames (list[dict])
            pc_labels (dict): Key: pc file name. Val: labels (list[dict])
                Each label includes following keys:
                    attributes, box, file_id, observation_angle, label_id
            im_labels (dict)
        """
        fname = os.path.join(self.timestamp_dir, seq_name, "frames_pc_laser.json")
        with open(fname, "r") as f:
            pc_frames = json.load(f)["data"]

        fname = os.path.join(self.timestamp_dir, seq_name, "frames_img_laser.json")
        with open(fname, "r") as f:
            im_frames = json.load(f)["data"]

        fname = os.path.join(self.pc_label_dir, f"{seq_name}.json")
        with open(fname, "r") as f:
            pc_labels = json.load(f)["labels"]

        fname = os.path.join(self.im_label_dir, f"{seq_name}.json")
        with open(fname, "r") as f:
            im_labels = json.load(f)["labels"]

        return pc_frames, im_frames, pc_labels, im_labels

    def _load_pointcloud(self, url):
        """Load a point cloud given file url.

        Returns:
            pc (np.ndarray[3, N]):
        """
        # pcd_load =
        # o3d.io.read_point_cloud(os.path.join(self.data_dir, url), format='pcd')
        # return np.asarray(pcd_load.points, dtype=np.float32)
        pc = point_cloud_from_path(os.path.join(self.data_dir, url)).pc_data
        # NOTE: redundent copy, ok for now
        pc = np.array([pc["x"], pc["y"], pc["z"]], dtype=np.float32)
        return pc

    def _load_consecutive_lasers(self, url):
        """Load current and previous consecutive laser scans.

        Args:
            url (str): file url of the current scan

        Returns:
            pc (np.ndarray[self.num_scan, N]): Forward in time with increasing
                row index, i.e. the latest scan is pc[-1]
        """
        fpath = os.path.dirname(url)
        a = os.path.basename(url).split(".")
        current_frame_idx = int(os.path.basename(url).split(".")[0])
        frames_list = []
        for del_idx in reversed(range(self.__num_scan)):
            frame_idx = max(0, current_frame_idx - del_idx * self.__scan_stride)
            url = os.path.join(fpath, str(frame_idx).zfill(6) + ".txt")
            frames_list.append(self._load_laser(url))

        return np.stack(frames_list, axis=0)

    def _load_laser(self, url):
        """Load a laser given file url.

        Returns:
            pc (np.ndarray[N, ]):
        """
        return np.loadtxt(os.path.join(self.data_dir, url), dtype=np.float32)

class PseudoDetection:
    def __init__(self, split, cfg, validation=False):
        assert split in ["train", "val", "test"], f'Invalid split "{split}"'
        self.radius_segment = cfg['radius_segment']
        self.perturb = cfg['perturb']

        data_dir = os.path.abspath(os.path.expanduser(cfg["data_dir"]))

        if split in ['train', 'val']:
            data_dir = os.path.join(data_dir, 'train' + "_dataset")
        else:
            data_dir = os.path.join(data_dir, split + "_dataset")

        self.data_dir = data_dir
        self.timestamp_dir = os.path.join(data_dir, "timestamps")
        self.pc_label_dir = os.path.join(data_dir, "labels", "labels_3d")

        self.sequence_names = (_JRDB_TRAIN_SEQUENCES if split == "train" else _JRDB_VAL_SEQUENCES)

        # self.sequence_names = os.listdir(self.timestamp_dir)
        self.sequence_pc_frames = []
        self.sequence_pc_labels = []

        self.__flat_inds_sequence = []
        self.__flat_inds_frame = []

        print('{} dataset: {} sequences selected'.format(split, len(self.sequence_names)))
        for idx, seq_name in enumerate(self.sequence_names[:1]): # Debugging, only use two sequences
            pc_frames, pc_labels = self._load_one_sequence(seq_name)
            self.sequence_pc_frames.append(pc_frames)
            self.sequence_pc_labels.append(pc_labels)

            # only use frames that has annotation
            pc_labeled_frames = []
            for fr_idx, fr in enumerate(pc_frames):
                fname = os.path.basename(fr["pointclouds"]["upper_velodyne"]["url"])
                if fname in pc_labels:
                    pc_labeled_frames.append(fr_idx)

            # build a flat index for all sequences and frames

            sequence_length = len(pc_labeled_frames)
            self.__flat_inds_sequence += sequence_length * [idx]
            self.__flat_inds_frame += pc_labeled_frames

    def __len__(self):
        return len(self.__flat_inds_frame)

    def __getitem__(self, idx):
        idx_sq = self.__flat_inds_sequence[idx]
        idx_fr = self.__flat_inds_frame[idx]

        pc_frame = self.sequence_pc_frames[idx_sq][idx_fr]

        laser_r = self._load_laser(pc_frame["laser"]["url"])
        laser_phi = np.linspace(-np.pi, np.pi, len(laser_r), dtype=np.float32)
        laser_x, laser_y= rphi_to_xy(laser_r, laser_phi)
        laser_z = -0.7 * np.ones(len(laser_r), dtype=np.float32)
        laser_xyz = jt.transform_pts_laser_to_base(np.stack((laser_x, laser_y, laser_z), axis=0)).T

        pc_fname = os.path.basename(pc_frame["pointclouds"]["upper_velodyne"]["url"])
        anns = self.sequence_pc_labels[idx_sq][pc_fname]

        segments, boxes, dets_center = self.anns_to_segments(laser_xyz[:, :2],
                                                             anns,
                                                             radius=self.radius_segment,
                                                             perturb=self.perturb) # Points in segments are transformed into base frame

        pc_frame.update(
            {
                "laser_r": laser_r,
                "laser_phi": laser_phi,
                "segments": segments,
                "boxes": boxes,
                "dets_center": dets_center,
                "laser_xyz": laser_xyz
            }
        )

        # #===============test===========
        #
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111)
        #
        # ax.cla()
        # ax.set_aspect("equal")
        # ax.set_xlim(-5, 5)
        # ax.set_ylim(-5, 5)
        #
        # for segment, box in zip(segments, boxes):
        #     ax.scatter(segment[..., 0], segment[..., 1], s=1, c='blue')
        #     c = plt.Circle(box.xyz[:2], radius=0.7, color='r', fill=False)
        #     ax.add_artist(c)
        #     box.draw_bev(ax, c='purple')
        #
        # plt.savefig("./tmp_imgs/dets/frame_%04d.png" % idx)
        # plt.close(fig)
        #
        # # ===============test===========


        return pc_frame

    def anns_to_segments(self, laser_xy, anns, radius=0.7, perturb=0.1):
        segments, boxes, dets_center = [], [], []
        for ann in anns:
            cx, cy = ann['box']['cx'], ann['box']['cy']  # Only consider xy plane
            alpha = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(-perturb, perturb)

            pesudo_center = np.array([cx + r * np.cos(alpha), cy + r * np.sin(alpha)])
            # pesudo_center = np.array([cx, cy])
            segment = laser_xy[np.linalg.norm(laser_xy - pesudo_center, axis=1) <= radius]
            segments.append(segment)
            boxes.append(jt.box_from_jrdb(ann["box"]))
            dets_center.append(pesudo_center)

        return segments, boxes, dets_center


    def _load_one_sequence(self, seq_name):
        """Load frame info and labels for one sequence.

        Args:
            seq_name (str): Sequence name

        Returns:
            pc_frames (list[dict]): Each element in the list contains information
                for the pointcloud at a single frame
            pc_labels (dict): Key: pc file name. Val: labels (list[dict])
                Each label includes following keys:
                    attributes, box, file_id, observation_angle, label_id
        """
        fname = os.path.join(self.timestamp_dir, seq_name, "frames_pc_laser.json")
        with open(fname, "r") as f:
            pc_frames = json.load(f)["data"]

        fname = os.path.join(self.pc_label_dir, f"{seq_name}.json")
        with open(fname, "r") as f:
            pc_labels = json.load(f)["labels"]

        return pc_frames, pc_labels

    def _load_laser(self, url):
        """Load a laser given file url.

        Returns:
            pc (np.ndarray[N, ]):
        """
        return np.loadtxt(os.path.join(self.data_dir, url), dtype=np.float32)

if __name__ == "__main__":
    dets = PseudoDetection(split='train', cfg={"data_dir": "./data/JRDB"})
    for i in range(1000):
        det = dets[i]
        continue
    print()
