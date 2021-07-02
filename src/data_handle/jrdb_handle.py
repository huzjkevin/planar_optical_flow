# from glob import glob
import json
import numpy as np
import os

from src.data_handle._pypcd import point_cloud_from_path
from src.utils.utils import rphi_to_xy
import src.utils.jrdb_transforms as jt
import matplotlib.pyplot as plt

# NOTE: Don't use open3d to load point cloud since it spams the console. Setting
# verbosity level does not solve the problem
# https://github.com/intel-isl/Open3D/issues/1921
# https://github.com/intel-isl/Open3D/issues/884

# Force the dataloader to load only one sample, in which case the network should
# fit perfectly.
_DEBUG_ONE_SAMPLE = False
_DEBUG = False


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

        if _DEBUG_ONE_SAMPLE:
            split = "train"

        # JRDB test set labels are not available
        if split == "test":
            split = "val"

        self.radius_segment = cfg["radius_segment"]
        self.perturb = cfg["perturb"]
        self.is_3d = cfg["is_3d"]

        data_dir = os.path.abspath(os.path.expanduser(cfg["data_dir"]))
        data_dir = os.path.join(data_dir, "train_dataset")

        self.data_dir = data_dir
        self.timestamp_dir = os.path.join(data_dir, "timestamps")
        self.pc_label_dir = os.path.join(data_dir, "labels", "labels_3d")

        self.sequence_names = (
            _JRDB_TRAIN_SEQUENCES if split == "train" else _JRDB_VAL_SEQUENCES
        )

        if _DEBUG:
            self.sequence_names = self.sequence_names[:1]  # for debugging

        print("{} dataset: {} sequences found".format(split, len(self.sequence_names)))

        self.sequence_pc_frames = []
        self.sequence_im_frames = []
        self.sequence_pc_labels = []
        self.sequence_im_labels = []
        self.__flat_inds_sequence = []
        self.__flat_inds_frame = []

        for idx, seq_name in enumerate(self.sequence_names):
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

        # NOTE It's important to use a copy as the return dict, otherwise the
        # original dict in the data handle will be corrupted
        pc_frame = self.sequence_pc_frames[idx_sq][idx_fr].copy()
        if self.is_3d:
            pc_data = self._load_pointcloud(
                pc_frame["pointclouds"]["upper_velodyne"]["url"]
            )
            points = jt.transform_pts_upper_velodyne_to_base(pc_data).T
        else:
            laser_r = self._load_laser(pc_frame["laser"]["url"])
            laser_phi = np.linspace(-np.pi, np.pi, len(laser_r), dtype=np.float32)
            laser_x, laser_y = rphi_to_xy(laser_r, laser_phi)
            laser_z = -0.7 * np.ones(len(laser_r), dtype=np.float32)
            points = jt.transform_pts_laser_to_base(
                np.stack((laser_x, laser_y, laser_z), axis=0)
            ).T

        pc_fname = os.path.basename(pc_frame["pointclouds"]["upper_velodyne"]["url"])
        anns = self.sequence_pc_labels[idx_sq][pc_fname]

        segments, boxes, dets_center = self.anns_to_segments(
            points, anns, radius=self.radius_segment, perturb=self.perturb
        )  # Points in segments are transformed into base frame

        pc_frame.update(
            {
                # "laser_r": laser_r,
                # "laser_phi": laser_phi,
                "segments": segments,
                "boxes": boxes,
                "dets_center": dets_center,
                "points": points,
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

    def anns_to_segments(self, points, anns, radius=0.7, perturb=0.1):
        """
        Generate input segments (set of laser points) according to the annotations of one frame

        :param points: (np.ndarray[N, 3]) laser points in xyz coordinates of one frame
        :param anns: list of annotations in one frame
        :param radius: (float) size of the region for selecting laser points into segements
        :param perturb: (float) threshold of the random perturbation of box center

        :return segments: List(S) list of segments of laser points used for input of the network. S is variable
        :return boxes: List(S) list of annotated bounding box
        :return dets_center: List(S) list of (pesudo) detection center
        """
        segments, boxes, dets_center = [], [], []
        if self.is_3d:
            for ann in anns:
                cx, cy, cz = (
                    ann["box"]["cx"],
                    ann["box"]["cy"],
                    ann["box"]["cz"],
                )  # Only consider xy plane
                alpha = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(-perturb, perturb)

                if _DEBUG_ONE_SAMPLE:
                    pesudo_center = np.array([cx, cy, 0.176])
                else:
                    pesudo_center = np.array(
                        [cx + r * np.cos(alpha), cy + r * np.sin(alpha), 0.176]
                    )

                # pesudo_center = np.array([cx, cy])
                # select points based on xy coordinate
                segment = points[
                    np.linalg.norm(points[:, :2] - pesudo_center[:2], axis=1) <= radius
                ]
                segments.append(segment)
                # boxes.append(jt.box_from_jrdb(ann["box"]))
                boxes.append(
                    np.array(
                        [
                            cx,
                            cy,
                            cz,
                            ann["box"]["l"],
                            ann["box"]["w"],
                            ann["box"]["h"],
                            ann["box"]["rot_z"],
                        ]
                    )
                )
                dets_center.append(pesudo_center)
        else:
            points = points[:, :2]
            for ann in anns:
                cx, cy = ann["box"]["cx"], ann["box"]["cy"]  # Only consider xy plane
                alpha = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(-perturb, perturb)

                if _DEBUG_ONE_SAMPLE:
                    pesudo_center = np.array([cx, cy])
                else:
                    pesudo_center = np.array(
                        [cx + r * np.cos(alpha), cy + r * np.sin(alpha)]
                    )

                # pesudo_center = np.array([cx, cy])

                segment = points[
                    np.linalg.norm(points - pesudo_center, axis=1) <= radius
                ]
                segments.append(segment)
                # boxes.append(jt.box_from_jrdb(ann["box"]))
                boxes.append(
                    cx, cy, ann["box"]["l"], ann["box"]["w"], ann["box"]["rot_z"]
                )
                dets_center.append(pesudo_center)

        return segments, np.array(boxes), np.array(dets_center)

    @staticmethod
    def box_is_on_ground(jrdb_ann_dict):
        bottom_h = float(jrdb_ann_dict["box"]["cz"]) - 0.5 * float(
            jrdb_ann_dict["box"]["h"]
        )

        return bottom_h < -0.69  # value found by examining dataset

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

    def _load_laser(self, url):
        """Load a laser scan given file url.

        Returns:
            laser scan (np.ndarray[N, 2]):
        """
        return np.loadtxt(os.path.join(self.data_dir, url), dtype=np.float32)


if __name__ == "__main__":
    dets = JRDBHandle(split="train", cfg={"data_dir": "./data/JRDB"})
    for i in range(1000):
        det = dets[i]
        continue
    print()
