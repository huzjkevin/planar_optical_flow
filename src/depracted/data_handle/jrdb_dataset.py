import numpy as np
from torch.utils.data import Dataset, DataLoader

# from .jrdb_handle import JRDBHandle, PseudoDetection
from src.data_handle.jrdb_handle import JRDBHandle, PseudoDetection # For debugging
import src.utils.utils as u
from scipy.spatial.distance import cdist
from src.utils.jrdb_transforms import Box3d
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

pi = np.pi

def get_dataloader(split, batch_size, num_workers, cfg):
    ds = JRDBDataset(split, cfg)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=ds.collate_batch,
    )
    return dl


class JRDBDataset(Dataset):
    def __init__(self, split, cfg):
        self.__handle = JRDBHandle(split, cfg["data_handle_kwargs"])

        self._augment_data = cfg["augment_data"]
        self._person_only = cfg["person_only"]
        self._cutout_kwargs = cfg["cutout_kwargs"]

    def __len__(self):
        return len(self.__handle)

    def __getitem__(self, idx):
        data_dict = self.__handle[idx]

        if self._augment_data:
            data_dict = u.data_augmentation(data_dict)

        data_dict["input"] = u.scans_to_cutout(
            data_dict["laser_data"],
            data_dict["laser_grid"],
            stride=1,
            **self._cutout_kwargs
        )

        return data_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            if k in ["target_cls", "target_reg", "input"]:
                rtn_dict[k] = np.array([sample[k] for sample in batch])
            else:
                rtn_dict[k] = [sample[k] for sample in batch]

        return rtn_dict

class JRDBPointNet(Dataset):
    def __init__(self, split, cfg, validation=False, min_num_pts=5, input_size=32):
        self.__handle = PseudoDetection(split, cfg=cfg, validation=validation)
        self.input_size = input_size

        self.inputs = []
        self.targets = []
        self.dets_center = []

        for frame in self.__handle:
            segments = frame['segments']
            boxes = frame['boxes']
            dets_center = frame['dets_center']

            for segment, box, det_center in zip(segments, boxes, dets_center):
                if len(segment) > min_num_pts:
                    self.inputs.append(np.array(segment))
                    target = np.array([box.xyz[0, 0], box.xyz[1, 0], box.lwh[0, 0], box.lwh[1, 0], box.rot_z])

                    if target[-1] > pi:
                        target[-1] -= 2 * pi
                    if target[-1] < -pi:
                        target[-1] += 2 * pi
                    # self.targets.append(np.hstack((target[:-1], np.cos(target[-1]))))
                    self.targets.append(target)
                    self.dets_center.append(det_center)

                    if cfg['use_data_augumentation'] and split == 'train':
                        input_aug, target_aug, det_center_aug = self.data_augumentation(np.array(segment),
                                                                                        target,
                                                                                        det_center)
                        self.inputs.append(input_aug)
                        # self.targets.append(np.hstack((target_aug[:-1], np.cos(target_aug[-1]))))
                        self.targets.append(target_aug)
                        self.dets_center.append(det_center_aug)



                        # # ===============test===========
                        # box_aug = Box3d(np.array([target_aug[0], target_aug[1], -0.7]), np.array([target_aug[2], target_aug[3], 0.5]), target_aug[4])
                        #
                        # fig = plt.figure(figsize=(10, 10))
                        # ax = fig.add_subplot(111)
                        #
                        # ax.cla()
                        # ax.set_aspect("equal")
                        # # ax.set_xlim(-5, 5)
                        # # ax.set_ylim(-5, 5)
                        #
                        # ax.scatter(self.inputs[-2][..., 0], self.inputs[-2][..., 1], s=1, c='red')
                        # c = plt.Circle(box.xyz[:2], radius=0.01, color='orange', fill=True)
                        # ax.add_artist(c)
                        # box.draw_bev(ax, c='red')
                        #
                        # ax.scatter(input_aug[..., 0], input_aug[..., 1], s=1, c='blue')
                        # c_aug = plt.Circle(box_aug.xyz[:2], radius=0.01, color='purple', fill=True)
                        # ax.add_artist(c_aug)
                        # box_aug.draw_bev(ax, c='blue')
                        #
                        # plt.savefig("./tmp_imgs/test/frame_%04d.png")
                        # plt.close(fig)
                        #
                        # # ===============test===========

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        idx = 200

        rtn_dict = {}

        input = self.inputs[idx]
        target = self.targets[idx]
        det_center = self.dets_center[idx]

        input = input - det_center
        target = target - np.hstack((det_center, [0, 0, 0]))
        target[-1] = target[-1] / pi

        #randomly drop 25% of the input points
        # np.random.shuffle(input)
        # input = input[:int(len(input) * 0.75)]

        # Interpolate the input data into fixed size
        if len(input) > self.input_size:
            np.random.shuffle(input)
            rtn_dict['input'] = input[:self.input_size]
        else:
            repeat = self.input_size // len(input)
            pad = self.input_size % len(input)
            np.random.shuffle(input)
            input = np.repeat(input, repeat, axis=0)
            input = np.vstack((input, input[:pad]))
            np.random.shuffle(input)
            rtn_dict['input'] = input

        rtn_dict['target'] = target
        rtn_dict['det_center'] = det_center

        return rtn_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            rtn_dict[k] = np.array([sample[k] for sample in batch])

        return rtn_dict

    def data_augumentation(self, input, target, det_center, rot_max=0.25*pi, dist_max=0.3):
        rot_z_rand = np.random.uniform(-rot_max, rot_max)
        box_center = target[:2]

        rot = u._phi_to_rotation_matrix(rot_z_rand)
        trans = np.random.uniform(-dist_max, dist_max, 2)

        input_aug = np.matmul(input - box_center, rot.T) + box_center + trans
        det_center_aug = np.matmul(det_center - box_center, rot.T) + box_center + trans
        box_center_aug = box_center + trans
        target_aug = np.hstack((box_center_aug, [target[2], target[3], target[-1] - rot_z_rand]))

        if target_aug[-1] > pi:
            target_aug[-1] -= 2*pi
        if target_aug[-1] < -pi:
            target_aug[-1] += 2*pi


        return input_aug, target_aug, det_center_aug


def _get_regression_target(
    scan,
    scan_phi,
    wcs,
    was,
    wps,
    radius_wc=0.6,
    radius_wa=0.4,
    radius_wp=0.35,
    label_wc=1,
    label_wa=2,
    label_wp=3,
    person_only=False,
):
    num_pts = len(scan)
    target_cls = np.zeros(num_pts, dtype=np.int64)
    target_reg = np.zeros((num_pts, 2), dtype=np.float32)

    if person_only:
        all_dets = list(wps)
        all_radius = [radius_wp] * len(wps)
        labels = [0] + [1] * len(wps)
    else:
        all_dets = list(wcs) + list(was) + list(wps)
        all_radius = (
            [radius_wc] * len(wcs) + [radius_wa] * len(was) + [radius_wp] * len(wps)
        )
        labels = (
            [0] + [label_wc] * len(wcs) + [label_wa] * len(was) + [label_wp] * len(wps)
        )

    dets = _closest_detection(scan, scan_phi, all_dets, all_radius)

    for i, (r, phi) in enumerate(zip(scan, scan_phi)):
        if 0 < dets[i]:
            target_cls[i] = labels[dets[i]]
            target_reg[i, :] = u.global_to_canonical(r, phi, *all_dets[dets[i] - 1])

    return target_cls, target_reg


def _closest_detection(scan, scan_phi, dets, radii):
    """
    Given a single `scan` (450 floats), a list of r,phi detections `dets` (Nx2),
    and a list of N `radii` for those detections, return a mapping from each
    point in `scan` to the closest detection for which the point falls inside its
    radius. The returned detection-index is a 1-based index, with 0 meaning no
    detection is close enough to that point.
    """
    if len(dets) == 0:
        return np.zeros_like(scan, dtype=int)

    assert len(dets) == len(radii), "Need to give a radius for each detection!"

    # Distance (in x,y space) of each laser-point with each detection.
    scan_xy = np.array(u.rphi_to_xy(scan, scan_phi)).T  # (N, 2)
    dists = cdist(scan_xy, np.array([u.rphi_to_xy(r, phi) for r, phi in dets]))

    # Subtract the radius from the distances, such that they are < 0 if inside,
    # > 0 if outside.
    dists -= radii

    # Prepend zeros so that argmin is 0 for everything "outside".
    dists = np.hstack([np.zeros((len(scan), 1)), dists])

    # And find out who's closest, including the threshold!
    return np.argmin(dists, axis=1)

if __name__ == '__main__':
    # Data statistics
    cfg = {
        "tag": "",
        "data_dir": "./data/JRDB",
        "epochs": 200,
        "batch_size": 256,
        "grad_norm_clip": 0.0,
        "num_workers": 8,
        "use_data_augumentation": False,
        "train_with_val": True,
        "use_polar_grid": False,
        "radius_segment": 0.4,
        "perturb": 0
    }

    split = 'train'

    dataset = JRDBPointNet(split=split, cfg=cfg)

    targets = np.array(dataset.targets)
    length = targets[..., 2]
    width = targets[..., 3]
    height = targets[..., 4]
    orientation = targets[..., 5]

    fig = plt.figure(figsize=(20, 25))
    ax_l = fig.add_subplot(411)
    ax_w = fig.add_subplot(412)
    ax_h = fig.add_subplot(413)
    ax_ori = fig.add_subplot(414)

    ax_l.set_title('{} length histogram. mean: {:0.3f}, var: {:0.3f}'.format(split, np.mean(length), np.var(length)))
    ax_w.set_title('{} width histogram. mean: {:0.3f}, var: {:0.3f}'.format(split, np.mean(width), np.var(width)))
    ax_h.set_title('{} height histogram. mean: {:0.3f}, var: {:0.3f}'.format(split, np.mean(height), np.var(height)))
    ax_ori.set_title('{} orientation histogram. mean: {:0.3f}, var: {:0.3f}'.format(split, np.mean(orientation), np.var(orientation)))

    n_bins = 200
    n_pts = len(targets)
    weights = np.ones(n_pts) / n_pts

    ax_l.hist(length, n_bins, weights=weights,
              histtype='bar',
              color='blue')

    ax_l.set_xlabel('length [m]')
    ax_l.yaxis.set_major_formatter(PercentFormatter(1))

    ax_w.hist(width, n_bins, weights=weights,
              histtype='bar',
              color='blue')

    ax_w.set_xlabel('width [m]')
    ax_w.yaxis.set_major_formatter(PercentFormatter(1))

    ax_h.hist(height, n_bins, weights=weights,
                histtype='bar',
                color='blue')

    ax_h.set_xlabel('height [m]')
    ax_h.yaxis.set_major_formatter(PercentFormatter(1))

    ax_ori.hist(orientation, n_bins, weights=weights,
                histtype='bar',
                color='blue')

    ax_ori.set_xlabel('orientation [rad]')
    ax_ori.yaxis.set_major_formatter(PercentFormatter(1))

    plt.savefig("./tmp_imgs/{}_data_statistics.png".format(split))

    plt.show()
    plt.close(fig)


    # for i in range(100):
    #     input = dataset[i]['input']
    #     target = dataset[i]['target']
    #     box = Box3d(np.array([target[0], target[1], -0.7]), np.array([target[2], target[3], 0.5]), target[4])
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
    #     ax.scatter(input[..., 0], input[..., 1], s=1, c='blue')
    #     c = plt.Circle(box.xyz[:2], radius=0.5, color='r', fill=False)
    #     ax.add_artist(c)
    #     box.draw_bev(ax, c='purple')
    #
    #     plt.savefig("./tmp_imgs/dets/frame_%04d.png" % i)
    #     plt.close(fig)
    #
    #     # ===============test===========
