from glob import glob
import os
import open3d as o3d
import json
from torch.utils.data import Dataset, DataLoader
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import matplotlib.pyplot as plt



class JRDBDataset(Dataset):
    def __init__(self, data_path, pcd_size=1024):
        seq_names = [d for d in os.listdir(os.path.join(data_path, 'pointclouds')) if os.path.isdir(os.path.join(data_path, 'pointclouds', d))]
        self.seq_names = seq_names
        self.pcd_size = pcd_size

        self.pt_clouds = [None] * len(self.seq_names)
        self.dets_ns = [None] * len(self.seq_names)
        self.dets = [None] * len(self.seq_names)
        self.pcds_ns = [None] * len(self.seq_names)

        self.flat_dets, self.inputs  = [], []
        self.__flat_seq_inds = []
        # self.__id2is = [None] * len(seq_names)

        for seq_idx, seq_name in enumerate(self.seq_names):

            pcd_files = sorted(glob(os.path.join(data_path, 'pointclouds', seq_name, '*.pcd')))
            self.pcds_ns[seq_idx] = pcd_files
            self.pt_clouds[seq_idx] = [self._load_pcd_file(pcd_file) for pcd_file in pcd_files]

            label_file = os.path.join(data_path, 'labels', seq_name + '.json')
            dets_ns, dets_box = self._load_det_file(label_file)
            self.dets_ns[seq_idx] = dets_ns
            self.dets[seq_idx] = dets_box

            dets_segment = self.pt_cloud_to_det_segments(self.pt_clouds[seq_idx], dets_ns, dets_box)

            for i in range(len(dets_box)):
                # num_samples = len(dets_box[i])
                # self.__flat_seq_inds += [seq_idx] * num_samples
                self.flat_dets += dets_box[i]
                self.inputs += dets_segment[i]
                self.__flat_seq_inds += [seq_idx] * len(dets_box)

            # # for debugging
            # sample_size = 100
            # self.flat_dets = self.flat_dets[:sample_size]
            # self.inputs = self.inputs[:sample_size]
            # self.__flat_seq_inds = self.__flat_seq_inds[:sample_size]

    def __len__(self):
        return len(self.flat_dets)

    def __getitem__(self, idx):
        # idx = 10

        rtn_dict = {}
        det = self.flat_dets[idx]
        input = self.inputs[idx]
        # seq_idx = self.__flat_seq_inds[idx]

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(input)
        # box_target = create_single_bounding_box(det)
        # # box_pred = create_single_bounding_box(x.data.cpu().numpy().reshape(-1))
        # box_target.color = np.array([1.0, 0.0, 0.0])
        # o3d.visualization.draw_geometries([pcd, box_target])

        if len(input) == 0:
            rtn_dict['input'], rtn_dict['target'] = None, None
            return rtn_dict

        if len(input) > self.pcd_size:
            rtn_dict['input'] = input[:self.pcd_size]
        else:
            repeat = self.pcd_size // len(input)
            pad = self.pcd_size % len(input)
            input = np.repeat(input, repeat, axis=0)
            input = np.vstack((input, input[:pad]))
            rtn_dict['input'] = input

        rtn_dict['target'] = det

        return rtn_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        batch_size = len(batch)
        inputs, targets = [], []

        for sample in batch:
            input, target = sample['input'], sample['target']
            if input is not None:
                inputs.append(input)
                targets.append(target)

        inputs = np.array(inputs)
        targets = np.array(targets)
        pad = batch_size - len(inputs)
        if pad > 0:
            inputs = np.vstack((inputs, inputs[:pad]))
            targets = np.vstack((targets, targets[:pad]))

        # #Test=======================================
        # for input, target in zip(inputs, targets):
        #
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(input)
        #     box = create_single_bounding_box(target)
        #     o3d.visualization.draw_geometries([pcd, box])
        #
        # # Test=======================================

        rtn_dict['input'] = inputs
        rtn_dict['target'] = targets

        return rtn_dict

    def pt_cloud_to_det_segments(self, pt_clouds, dets_ns, dets):
        dets_segment = []

        # Iterate over all frames
        for det_ns, det_list in zip(dets_ns, dets):
            pcd_idx = int(det_ns[:-4])
            pt_cloud = pt_clouds[pcd_idx]

            # Iterate over all detection of each frame
            det_segment_list = []
            for det in det_list:
                det_segment, mask = self.find_points_inside_bounding_box(pt_cloud, det)
                # if len(det_segment) > 0:
                det_segment_list.append(det_segment)

            dets_segment.append(det_segment_list)

            # for segment, det in zip(det_segment_list, det_list):
            #     #Test=======================================
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(pt_cloud)
            #
            #     seg = o3d.geometry.PointCloud()
            #     seg.points = o3d.utility.Vector3dVector(segment)
            #     colors = [[1, 0, 0] for i in range(len(segment))]
            #     seg.colors = o3d.utility.Vector3dVector(colors)
            #
            #     box = create_single_bounding_box(det)
            #     print('det: ', det)
            #     print("center: ", np.mean(segment, axis=0))
            #     o3d.visualization.draw_geometries([pcd, box, seg])
            #
            #     # Test=======================================

        return dets_segment



    def find_points_inside_bounding_box(self, pt_cloud, det):
        cx, cy, cz, l, w, h, rot_z = det
        sinz = np.sin(rot_z)
        cosz = np.cos(rot_z)
        rot = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        trans = np.array([cx, cy, cz]).reshape(1, -1)

        #transform pt_cloud into bounding box frame
        pt_cloud_box = np.matmul(pt_cloud - trans, rot) #- np.matmul(trans, rot)

        mask_x = np.logical_and(pt_cloud_box[..., 0] > -w / 2, pt_cloud_box[..., 0] < w / 2)
        mask_y = np.logical_and(pt_cloud_box[..., 1] > -l / 2, pt_cloud_box[..., 1] < l / 2)
        mask_z = np.logical_and(pt_cloud_box[..., 2] > -h / 2, pt_cloud_box[..., 2] < h / 2)

        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

        # #Test=======================================
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pt_cloud)
        #
        # a = pt_cloud[mask]
        # seg = o3d.geometry.PointCloud()
        # seg.points = o3d.utility.Vector3dVector(pt_cloud[mask])
        # colors = [[1, 0, 0] for i in range(len(pt_cloud[mask]))]
        # seg.colors = o3d.utility.Vector3dVector(colors)
        #
        # box = create_single_bounding_box(det)
        # o3d.visualization.draw_geometries([pcd, box, seg])
        #
        # if len(a) == 0:
        #     print(a)
        #
        # # Test=======================================

        return pt_cloud[mask], mask


    def _load_pcd_file(self, path):
        pcd = o3d.io.read_point_cloud(path)
        pcd = np.asarray(pcd.points)
        # pcd[..., 2] = 0.0

        return pcd

    def _load_det_file(self, path):
        dets_ns , dets_box = [], []
        with open(path) as f:
            json_data = json.load(f)
            for frame_idx, dets_list in json_data['labels'].items():
                dets_ns.append(frame_idx)
                # dets_box.append([[det['box']['cx'], det['box']['cy'], det['box']['l'], det['box']['w'], det['box']['rot_z']] for det in dets_list])
                dets_box.append([[det['box']['cx'],
                                  det['box']['cy'],
                                  det['box']['cz'],
                                  det['box']['l'],
                                  det['box']['w'],
                                  det['box']['h'],
                                  det['box']['rot_z']] for det in dets_list])

        return dets_ns, dets_box

def point_cloud_viz(pt_clouds, dets):
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(pt_clouds[0])
    boxes = create_bounding_box(dets[0])
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(geometry)

    for box in boxes:
        vis.add_geometry(box)

    for pt_cloud, det in zip(pt_clouds, dets):
        geometry.points = o3d.utility.Vector3dVector(pt_cloud)
        boxes = create_bounding_box(det)

        vis.update_geometry(geometry)
        for box in boxes:
            vis.update_geometry(box)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1 / 20)

    vis.destroy_window()

def create_bounding_box(dets):
    boxes = []
    for det in dets:
        cx, cy, cz, l, w, h, rot_z = det
        center = np.array([cx, cy, cz]).reshape(-1, 1)
        extent = np.array([l, w, h])
        vertices = []
        offset_x = np.array([-l, l])
        offset_y = np.array([-w, w])
        offset_z = np.array([-h, h])

        sinz = np.sin(rot_z)
        cosz = np.cos(rot_z)
        rot = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])

        for x in offset_x:
            for y in offset_y:
                for z in offset_z:
                    vertices.append([x, y, z])

        vertices = np.array(vertices)
        vertices = np.matmul(vertices, rot.T)

        lines = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]

        # colors = [[1, 0, 0] for i in range(len(lines))]
        # line_set = o3d.geometry.LineSet(
        #     points=o3d.utility.Vector3dVector(vertices),
        #     lines=o3d.utility.Vector2iVector(lines),
        # )
        # line_set.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([line_set])

        # bounding_box = o3d.geometry.PointCloud()
        # bounding_box.points = o3d.utility.Vector3dVector(vertices)
        # boxes.append(line_set)

        box = o3d.geometry.OrientedBoundingBox(center, rot, extent)
        boxes.append(box)

    return boxes

def create_single_bounding_box(box_param):
    cx, cy, cz, l, w, h, rot_z = box_param
    center = np.array([cx, cy, cz]).reshape(-1, 1)
    extent = np.array([l, w, h])

    sinz = np.sin(rot_z)
    cosz = np.cos(rot_z)
    rot = np.array([[cosz, -sinz, 0],
                    [sinz, cosz, 0],
                    [0, 0, 1]])

    box = o3d.geometry.OrientedBoundingBox(center, rot, extent)

    return box

if __name__ == '__main__':

    dataset = JRDBDataset(data_path='/media/kevin/Kevin_Linux/data_backup/CV_data/JRDB_sample')

    video_path = '../../../../tmp_videos/point_cloud_sample.mp4'

    # point_cloud_viz(dataset.pt_clouds[0], dataset.dets[0])


    fig = plt.figure(figsize=(8, 6))
    canvas = FigureCanvas(fig)
    w, h = canvas.get_width_height()
    ax = fig.add_subplot(111, projection='3d')
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w, h))

    for i in range(len(dataset)):
        input, target = dataset[i]['input'], dataset[i]['target']

        if input is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(input)
            box = create_single_bounding_box(target)
            o3d.visualization.draw_geometries([pcd, box])

    #     ax.cla()
    #
    #     # ax.set_aspect('equal')
    #     ax.set_xlim(-10, 10)
    #     ax.set_ylim(-10, 10)
    #     ax.set_zlim(-1, 5)
    #
    #     # ax.set_title('Frame: {}'.format(key[:-4]))
    #     # ax.axis("off")
    #
    #     ax.scatter(input[..., 0], input[..., 1], input[..., 2], s=1, c='blue')
    #     ax.scatter(target[0], target[1], target[2], s=5, color='red')
    #
    #     canvas.draw()
    #     img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, -1)
    #     bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     writer.write(bgr)
    #
    # plt.close(fig)
    # writer.release()