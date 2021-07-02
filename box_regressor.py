import torch
import numpy as np
from src.model.box_regression import BoundingBoxRegressor
from src.data_handle.jrdb_handle import JRDBHandle
from src.utils.rotate_iou import rotate_iou_gpu_eval
import os


class BoxRegressor(object):
    """
    The class for box regression. It can regress parameters of both 2d and 3d boxes.
    The default is 2d box regression.
    To regress 3d box, set argument 'is_3d' to True when createing the object.
    """

    def __init__(self, ckpt, gpu=True, is_3d=False):
        self.cfg = {
            "model": {
                "type": "box_reg",
                "input_dim": 3,
                "target_dim": 3,
                "dropout": 0.3,
            },
            "segment_radius": 0.4,
            "result_dir": "./output_box_reg",
            "min_segment_size": 5,
            "input_size": 64,
        }
        self.is_3d = is_3d
        self.gpu = gpu

        if self.is_3d:
            self.cfg["model"]["input_dim"] = 4
            self.cfg["model"]["target_dim"] = 5

        model = BoundingBoxRegressor(self.cfg["model"])
        model.load_state_dict(torch.load(ckpt)["model_state"])
        self.model = model
        self.model.eval()
        if gpu:
            self.model.cuda()

    def __call__(self, points, det_center, det_ori):
        """
        regress the bounding box's parameters (dims, orientation_residual) of one detection (given detection center)

        :param points: (np.ndarray[N, 2] / np.ndarray[N, 3]) points in xy(z) coordinates of one frame
        :param det_center: (np.ndarray[2,] / np.ndarray[3,]) the detection center used for selecting input points of the box regressor
        :param det_ori: (float) the predicted orientation from dr-spaam

        :return: (np.ndarray[5,] / np.ndarray[7,]) box parameter, i.e., [cx, cy, l, w, rot_z], where rot_z = det_ori + pred[-1] (ori_residual)
        """
        # preprocess
        input = self.generate_segment(points, det_center, self.cfg["segment_radius"])

        # discard invalid segment size
        if len(input) < self.cfg["min_segment_size"]:
            return None

        # Interpolate the input data into fixed size
        if len(input) > self.cfg["input_size"]:
            np.random.shuffle(input)
            input = input[: self.cfg["input_size"]]
        else:
            repeat = self.cfg["input_size"] // len(input)
            pad = self.cfg["input_size"] % len(input)
            np.random.shuffle(input)
            input = np.repeat(input, repeat, axis=0)
            input = np.vstack((input, input[:pad]))
            np.random.shuffle(input)

        input = input - det_center  # transform to canonical
        input = np.hstack((input, np.repeat(det_ori, len(input)).reshape(-1, 1)))
        input = np.expand_dims(input, axis=0)  # input.shape = (1, n_pts, 3)

        if self.gpu:
            input = torch.from_numpy(input).cuda(non_blocking=True).float()

        # if self.is_3d:
        #     det_center = np.append(det_center, 0.126)  # 0.126 is the mean of cz

        pred = self.model(input)[0]  # output channel: (length, width, ori_residual)
        pred = pred.data.cpu().numpy()

        if self.is_3d:
            pred[0] += det_center[-1]  # transform cz to global

        pred = np.hstack((det_center[:2], pred))
        pred[-1] = pred[-1] + det_ori  # rot_z = det_ori + ori_residual

        # pred = [cx, cy, l, w, rot_z] or [cx, cy, cz, l, w, h, rot_z] if is_3d is True
        return pred

    def generate_segment(self, points, det_center, radius=0.4):
        """
        Generate input segment (set of laser points) according to the detection center

        :param points: (np.ndarray[N, 2] / np.ndarray[N, 3]) points in xy(z) coordinates of one frame
        :param det_center: (np.ndarray[1, 2] / np.ndarray[1, 3]) the detection center used for selecting input points of the box regressor
        :param radius: (float) size of the region for selecting points into segements

        :return segment: (np.ndarray[S, 2] / np.ndarray[S, 3]) segment as a set of points used for input of the regressor. S is variable
        """

        return points[np.linalg.norm(points - det_center, axis=1) <= radius]


if __name__ == "__main__":
    split = "val"
    is_3d = False
    dataset_cfg = {
        "data_dir": "./data/JRDB",
        "radius_segment": 0.4,
        "perturb": 0.1,
        "is_3d": is_3d,
    }

    if is_3d:
        ckpt = "./pre_trained_ckpts/3d/ckpt_e100.pth"
    else:
        ckpt = "./pre_trained_ckpts/2d/ckpt_e100.pth"

    dataset = JRDBHandle(split, dataset_cfg)
    box_regressor = BoxRegressor(ckpt, is_3d=is_3d)

    ious, locs_loss, dims_loss, oris_loss = [], [], [], []

    for data in dataset:
        if is_3d:
            points = data["points"]
        else:
            points = data["points"][:, :2]
        dets_center = data["dets_center"]
        targets = data["boxes"]
        for det_center, target in zip(dets_center, targets):

            # if is_3d:
            #     target = np.array(
            #         [
            #             box_target.xyz[0, 0],
            #             box_target.xyz[1, 0],
            #             box_target.xyz[2, 0],
            #             box_target.lwh[0, 0],
            #             box_target.lwh[1, 0],
            #             box_target.lwh[2, 0],
            #             box_target.rot_z,
            #         ]
            #     )
            #     det_center = np.append(det_center, 0.126)  # 0.126 is the mean of cz
            # else:
            #     target = np.array(
            #         [
            #             box_target.xyz[0, 0],
            #             box_target.xyz[1, 0],
            #             box_target.lwh[0, 0],
            #             box_target.lwh[1, 0],
            #             box_target.rot_z,
            #         ]
            #     )
            if target[-1] > np.pi:
                target[-1] -= 2 * np.pi
            if target[-1] < -np.pi:
                target[-1] += 2 * np.pi

            # Pesudo orientation obtained from the tracking network
            det_ori = target[-1] + np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)

            # inference
            pred = box_regressor(points, det_center, det_ori)

            if pred is None:
                continue

            iou = rotate_iou_gpu_eval(
                pred.reshape(1, -1), target.reshape(1, -1), is_3d=is_3d
            )[0, 0]
            if is_3d:
                location_err = np.sum(np.linalg.norm(pred[:3] - target[:3]))
                dimension_err = np.sum(np.abs(pred[3:5] - target[3:5]))
                orientation_err = np.abs(pred[-1] - target[-1])
            else:
                location_err = np.sum(np.linalg.norm(pred[:2] - target[:2]))
                dimension_err = np.sum(np.abs(pred[2:4] - target[2:4]))
                orientation_err = np.abs(pred[-1] - target[-1])

            ious.append(iou)
            locs_loss.append(location_err)
            dims_loss.append(dimension_err)
            oris_loss.append(orientation_err)

    print("avg iou: ", np.mean(ious))
    print("avg loc_loss: ", np.mean(locs_loss))
    print("avg dim_loss: ", np.mean(dims_loss))
    print("avg ori_loss: ", np.mean(oris_loss))
