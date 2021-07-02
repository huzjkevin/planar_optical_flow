import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.data_handle.jrdb_handle import JRDBHandle
import src.utils.utils as u
from scipy.spatial.distance import cdist
from src.utils.jrdb_transforms import Box3d
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Force the dataloader to load only one sample, in which case the network should
# fit perfectly.
_DEBUG_ONE_SAMPLE = False
_INPUT_WITH_ANGLE = True
pi = np.pi


class JRDBBoxRegressionDataset(Dataset):
    """
    Dataset class for box regression
    """

    def __init__(self, split, cfg):
        self.__handle = JRDBHandle(split, cfg=cfg)
        self.input_size = cfg["input_size"]
        self.is_3d = cfg["is_3d"]
        self.mode = split  # train, val, test mode

        self.inputs = []
        self.targets = []
        self.targets_neighbor = (
            []
        )  # keep track of annotations nearby, for computing iou
        self.dets_center = []
        self.augmentation_kwargs = cfg["augmentation_kwargs"]

        for frame in self.__handle:
            segments = frame["segments"]
            boxes = frame["boxes"]
            dets_center = frame["dets_center"]

            for segment, box, det_center in zip(segments, boxes, dets_center):
                if len(segment) > cfg["min_segment_size"]:
                    self.inputs.append(np.array(segment))
                    # if self.is_3d:
                    #     target = np.array(
                    #         [
                    #             box.xyz[0, 0],
                    #             box.xyz[1, 0],
                    #             box.xyz[2, 0],
                    #             box.lwh[0, 0],
                    #             box.lwh[1, 0],
                    #             box.lwh[2, 0],
                    #             box.rot_z,
                    #         ]
                    #     )
                    #     # det_center = np.append(
                    #     #     det_center, 0.126
                    #     # )  # 0.126 is the mean of cz
                    # else:
                    #     target = np.array(
                    #         [
                    #             box.xyz[0, 0],
                    #             box.xyz[1, 0],
                    #             box.lwh[0, 0],
                    #             box.lwh[1, 0],
                    #             box.rot_z,
                    #         ]
                    #     box

                    if box[-1] > pi:
                        box[-1] -= 2 * pi
                    if box[-1] < -pi:
                        box[-1] += 2 * pi

                    self.targets.append(box)
                    self.targets_neighbor.append(
                        self.get_nearby_annotations(box, boxes)
                    )
                    self.dets_center.append(det_center)

                    if (
                        self.augmentation_kwargs["use_data_augmentation"]
                        and split == "train"
                    ):
                        input_aug, target_aug, det_center_aug = self.data_augmentation(
                            np.array(segment), box, det_center
                        )
                        self.inputs.append(input_aug)
                        self.targets.append(target_aug)
                        self.targets_neighbor.append(
                            self.get_nearby_annotations(target_aug, boxes)
                        )
                        self.dets_center.append(det_center_aug)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if _DEBUG_ONE_SAMPLE:
            idx = 20

        rtn_dict = {}

        input = self.inputs[idx]
        det_center = self.dets_center[idx]
        target_neighbor = self.targets_neighbor[idx]

        if self.is_3d:
            target = self.targets[idx][2:]  # cz, dims, rot_z
            box_center = self.targets[idx][:3].copy()
        else:
            target = self.targets[idx][2:]
            box_center = self.targets[idx][:2].copy()

        # transform to canonical
        input = input - det_center
        target[0] = target[0] - det_center[-1]
        # box_center[:2] -= det_center
        # target[-1] = target[-1] / pi
        if _INPUT_WITH_ANGLE:
            rot_z = target[-1]
            rtn_dict["rot_z"] = rot_z
            input_angle = rot_z + np.random.uniform(
                -self.augmentation_kwargs["rot_max"] * pi,
                self.augmentation_kwargs["rot_max"] * pi,
            )
            input = np.hstack(
                (input, np.repeat(input_angle, len(input)).reshape(-1, 1))
            )
            target[-1] = rot_z - input_angle

        # randomly drop  the input points
        if self.augmentation_kwargs["use_data_augmentation"] and self.mode == "train":
            np.random.shuffle(input)
            input = input[int(len(input) * self.augmentation_kwargs["random_drop"]) :]

        # Interpolate the input data into fixed size
        if len(input) > self.input_size:
            np.random.shuffle(input)
            rtn_dict["input"] = input[: self.input_size]
        else:
            repeat = self.input_size // len(input)
            pad = self.input_size % len(input)
            np.random.shuffle(input)
            input = np.repeat(input, repeat, axis=0)
            input = np.vstack((input, input[:pad]))
            np.random.shuffle(input)
            rtn_dict["input"] = input

        rtn_dict["target"] = target
        rtn_dict["det_center"] = det_center
        rtn_dict["box_center"] = box_center
        rtn_dict["target_neighbor"] = target_neighbor

        return rtn_dict

    def data_augmentation(self, input, target, det_center):
        """
        Augment input, target and detection center by randomly rotation and translation

        :param input: (np.ndarray[N, 2]) input segment
        :param target: (np.ndarray[5,] / (np.ndarray[7,])) ground truth box parameters: center, dims, rot_z
        :param det_center: detection center

        :return: augmented data, i.e., input_aug, target_aug, det_center_aug
        """
        rot_z_rand = np.random.uniform(
            -self.augmentation_kwargs["rot_max"] * pi,
            self.augmentation_kwargs["rot_max"] * pi,
        )
        dim_rand = 1.0 + np.random.uniform(
            -self.augmentation_kwargs["dim_max"], self.augmentation_kwargs["dim_max"]
        )
        trans = np.random.uniform(
            -self.augmentation_kwargs["dist_max"],
            self.augmentation_kwargs["dist_max"],
            2,
        )

        if self.is_3d:
            box_center = target[:2]
            rot = u._phi_to_rotation_matrix(rot_z_rand)
            input_aug = input.copy()
            input_aug[:, :2] = (
                np.matmul(input[:, :2] - box_center, rot.T) + box_center + trans
            )
            det_center_aug = (
                np.matmul(det_center[:2] - box_center, rot.T) + box_center + trans
            )
            box_center_aug = box_center + trans
            det_center_aug = np.append(det_center_aug, det_center[-1])
            box_center_aug = np.append(box_center_aug, target[2])
            target_aug = np.hstack(
                (
                    box_center_aug,
                    [
                        target[3] * dim_rand,
                        target[4] * dim_rand,
                        target[5] * dim_rand,
                        target[-1] - rot_z_rand,
                    ],
                )
            )
        else:
            box_center = target[:2]
            rot = u._phi_to_rotation_matrix(rot_z_rand)

            input_aug = np.matmul(input - box_center, rot.T) + box_center + trans
            det_center_aug = (
                np.matmul(det_center - box_center, rot.T) + box_center + trans
            )
            box_center_aug = box_center + trans
            target_aug = np.hstack(
                (
                    box_center_aug,
                    [
                        target[2] * dim_rand,
                        target[3] * dim_rand,
                        target[-1] - rot_z_rand,
                    ],
                )
            )

        if target_aug[-1] > pi:
            target_aug[-1] -= 2 * pi
        if target_aug[-1] < -pi:
            target_aug[-1] += 2 * pi

        return input_aug, target_aug, det_center_aug

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            rtn_dict[k] = np.array([sample[k] for sample in batch])

        return rtn_dict

    def get_nearby_annotations(self, target, anns, radius=1.0):
        nearby_anns = anns[np.linalg.norm(anns[:, :3] - target[:3], axis=1) <= radius]
        return np.append(nearby_anns, target.reshape(1, -1), axis=0)


if __name__ == "__main__":
    # Data statistics
    cfg = {
        "tag": "",
        "epochs": 200,
        "batch_size": 256,
        "grad_norm_clip": 0.0,
        "num_workers": 8,
        "dataset": {
            "data_dir": "./data/JRDB",
            "radius_segment": 0.4,
            "perturb": 0.1,
            "train_with_val": True,
            "is_3d": True,
            "min_segment_size": 5,
            "input_size": 32,
            "augmentation_kwargs": {
                "use_data_augmentation": False,
                "rot_max": 0.25,
                "dist_max": 0.3,
                "random_drop": 0.25,
            },
        },
    }

    split = "train"

    dataset = JRDBBoxRegressionDataset(split=split, cfg=cfg["dataset"])

    targets = np.array(dataset.targets)
    # length = targets[..., 2]
    # width = targets[..., 3]
    cz = targets[..., 2]
    print("z mean / var: {} / {}".format(np.mean(cz), np.var(cz)))
    # orientation = targets[..., 5]
    #
    # fig = plt.figure(figsize=(20, 25))
    # ax_l = fig.add_subplot(411)
    # ax_w = fig.add_subplot(412)
    # ax_h = fig.add_subplot(413)
    # ax_ori = fig.add_subplot(414)
    #
    # ax_l.set_title('{} length histogram. mean: {:0.3f}, var: {:0.3f}'.format(split, np.mean(length), np.var(length)))
    # ax_w.set_title('{} width histogram. mean: {:0.3f}, var: {:0.3f}'.format(split, np.mean(width), np.var(width)))
    # ax_h.set_title('{} height histogram. mean: {:0.3f}, var: {:0.3f}'.format(split, np.mean(height), np.var(height)))
    # ax_ori.set_title('{} orientation histogram. mean: {:0.3f}, var: {:0.3f}'.format(split, np.mean(orientation), np.var(orientation)))
    #
    # n_bins = 200
    # n_pts = len(targets)
    # weights = np.ones(n_pts) / n_pts
    #
    # ax_l.hist(length, n_bins, weights=weights,
    #           histtype='bar',
    #           color='blue')
    #
    # ax_l.set_xlabel('length [m]')
    # ax_l.yaxis.set_major_formatter(PercentFormatter(1))
    #
    # ax_w.hist(width, n_bins, weights=weights,
    #           histtype='bar',
    #           color='blue')
    #
    # ax_w.set_xlabel('width [m]')
    # ax_w.yaxis.set_major_formatter(PercentFormatter(1))
    #
    # ax_h.hist(height, n_bins, weights=weights,
    #             histtype='bar',
    #             color='blue')
    #
    # ax_h.set_xlabel('height [m]')
    # ax_h.yaxis.set_major_formatter(PercentFormatter(1))
    #
    # ax_ori.hist(orientation, n_bins, weights=weights,
    #             histtype='bar',
    #             color='blue')
    #
    # ax_ori.set_xlabel('orientation [rad]')
    # ax_ori.yaxis.set_major_formatter(PercentFormatter(1))
    #
    # plt.savefig("./tmp_imgs/{}_data_statistics.png".format(split))
    #
    # plt.show()
    # plt.close(fig)

    # for i in range(100):
    #     input = dataset[i]["input"]
    #     target = dataset[i]["target"]
    #     box = Box3d(
    #         np.array([target[0], target[1], -0.7]),
    #         np.array([target[2], target[3], 0.5]),
    #         target[4],
    #     )
    #
    #     # ===============test===========
    #
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(111)
    #
    #     ax.cla()
    #     ax.set_aspect("equal")
    #     ax.set_xlim(-5, 5)
    #     ax.set_ylim(-5, 5)
    #
    #     ax.scatter(input[..., 0], input[..., 1], s=1, c="blue")
    #     c = plt.Circle(box.xyz[:2], radius=0.5, color="r", fill=False)
    #     ax.add_artist(c)
    #     box.draw_bev(ax, c="purple")
    #
    #     plt.savefig("./tmp_imgs/test/frame_%04d.png" % i)
    #     plt.close(fig)
    #
    #     # ===============test===========
