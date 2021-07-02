from src.data_handle.depracted.drow_handle import DROWHandle
import src.utils.utils as u
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
import argparse
import yaml
import os
from sklearn import linear_model

def nms_predicted_center(data, preds, scores, min_dist=1.0):
    # data = [[segment0, label0], ...]
    # Sort predictions and data in descending order
    sort_ids = np.argsort(preds)[::-1]
    preds = preds[sort_ids]
    scores = scores[sort_ids]
    segments = [data[i][0] for i in sort_ids]

    # compute central positions of segments
    segment_centers = np.array([np.mean(segment, axis=0) for segment in segments])

    # compute pair-wise distance
    num_pts = len(segment_centers)
    xdiff = segment_centers[..., 0].reshape(num_pts, 1) - segment_centers[..., 0].reshape(1, num_pts)
    ydiff = segment_centers[..., 1].reshape(num_pts, 1) - segment_centers[..., 1].reshape(1, num_pts)
    p_dist = np.sqrt(np.square(xdiff) + np.square(ydiff))

    # nms
    for i in range(num_pts):
        if scores[i] <= 0.0:
            continue
        score = scores[i]
        dup_inds = p_dist[i] < min_dist
        scores[dup_inds] = 0.0
        scores[i] = score

    return segments, preds, scores

class Dataset:
    def __init__(self, split, cfg):
        self.__handle = DROWHandle(split, cfg['DROWHandle'])
        self.scans_data = []
        self.dets_wp = []
        for idx in range(len(self.__handle)):
            data_dict = self.__handle[idx]
            scan = data_dict['scans'][-1]
            dets_wp = data_dict['dets_wc']
            scan_phi = data_dict['scan_phi']
            odom_t = data_dict['odom_t']
            self.dets_wp.append(dets_wp)

            segments, labels, cut_ids = self.scan_to_segments(scan, scan_phi, dets_wp)
            self.scans_data.append([[], cut_ids, scan, odom_t])
            for segment, label in zip(segments, labels):
                if len(segment) > 2:
                    self.scans_data[idx][0].append([segment, label])

        self.scans_feature, self.labels = self._collate_data()
        self.input, self.target = np.vstack(self.scans_feature), np.hstack(self.labels)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        data = self.scans_data[idx]

        feature = self.compute_feature(data, idx)

        return feature

    def scan_to_segments(self, scan, scan_phi, wps, radius_wp=0.5, positive_thresh=0.5, jump_dist=0.5):
        scan_xy = np.array(u.rphi_to_xy(scan, scan_phi)).T
        pw_labels = np.zeros_like(scan)
        all_dets = np.array(wps)
        all_radius = np.array([radius_wp] * len(wps))

        cut_ids = np.clip(np.where(np.abs((scan[1:] - scan[:-1])) >= jump_dist)[0] + 1, 0, len(scan) - 1)
        segments = np.split(scan_xy, cut_ids, axis=0)

        for wp in wps:
            dist = np.linalg.norm(scan_xy - wp, axis=-1)
            pw_labels[dist <= radius_wp] = 1.0

        labels = -1.0 * np.ones(len(segments))
        for idx, segment in enumerate(segments):
            dist = np.array([np.linalg.norm(np.mean(segment, axis=0) - wp) for wp in wps])
            if np.any(dist <= radius_wp):
                labels[idx] = 1.0

        return segments, labels, cut_ids

    def _collate_data(self):
        X, Y = [], []
        for idx, data in enumerate(self.scans_data):
            feature = np.array(self.compute_feature(data, idx))
            X.append(feature[:, :-1])
            Y.append(feature[:, -1])

        return X, Y


    def compute_feature(self, data, scan_ids):
        scan_feature = []
        segments = data[0]
        cut_ids = data[1]
        curr_scan = data[2]
        curr_segments = np.split(curr_scan, cut_ids, axis=0)
        odom = data[-1]
        next_scan = self.scans_data[min(scan_ids + 1, len(self.scans_data) - 1)][2]
        next_segments = np.split(next_scan, cut_ids, axis=0)
        next_odom = self.scans_data[min(scan_ids + 1, len(self.scans_data) - 1)][-1]

        for idx, (segment, label) in enumerate(segments):
            # segment, label = segment[0], segment[1]
            seg_feature = []

            # number of points of segment
            n = len(segment)
            seg_feature.append(n)

            #Standard deviation
            mean = np.mean(segment, axis=0)
            dist = np.linalg.norm(segment - mean, axis=-1)
            sigma = np.sqrt(np.sum(np.square(dist))) / (n - 1)
            seg_feature.append(sigma)

            #Mean average deviation from median
            median = np.median(segment, axis=0)
            median_sigma = np.sum(np.linalg.norm(segment - median), axis=-1) / n
            seg_feature.append(median_sigma)

            #Jump distance preceedding / succeeding
            segment_prevs = segments[max(0, idx - 1)][0] # [segment, label]
            segment_next = segments[min(idx + 1, len(data) - 1)][0]
            jump_dist_preceeding = np.linalg.norm(segment_prevs[-1] - segment[0])
            jump_dist_succeeding = np.linalg.norm(segment[-1] - segment_next[0])
            seg_feature.append(jump_dist_preceeding)
            seg_feature.append(jump_dist_succeeding)

            #Width
            width = np.linalg.norm(segment[-1] - segment[0])
            seg_feature.append(width)

            #Linearity
            regressor = linear_model.LinearRegression()
            regressor.fit(segment[..., 0].reshape(-1, 1), segment[..., 1].reshape(-1, 1))

            # y = kx + b -> ax + by + c = 0
            a = regressor.coef_
            b = -1.0
            c = regressor.intercept_

            # ax + by + c = 0 -> x*cos(alpha) + y*sin(alpha) - r = 0
            cos_alpha = a / np.sqrt(a**2 + b**2)
            sin_alpha = b / np.sqrt(a**2 + b**2)
            r = np.abs(c / np.sqrt(a**2 + b**2))

            residual = np.sum(segment[..., 0] * cos_alpha + segment[..., 1] * sin_alpha - r)
            seg_feature.append(residual)

            #Circularity
            A = np.hstack((-2.0 * segment, np.ones((n, 1))))
            b = - np.square(segment[..., 0]) - np.square(segment[..., 1])
            X = np.matmul(np.linalg.pinv(A), b)
            xc, yc = X[0], X[1]
            rc = np.sqrt(xc**2 + yc**2 - X[2])
            Sc = np.sum(np.square(rc - np.sqrt(np.linalg.norm(X[:-1] - segment, axis=-1))))
            seg_feature.append(Sc)

            #Radius
            seg_feature.append(rc)

            #Boundary length
            boundary_len = np.sum(np.linalg.norm(segment[1:] - segment[:-1], axis=-1))
            seg_feature.append(boundary_len)

            #Boundary reglarity
            boundary_reg = np.std(np.linalg.norm(segment[1:] - segment[:-1], axis=-1))
            seg_feature.append(boundary_reg)

            #Mean curvature
            ptsA = segment[:-2]
            ptsB = segment[1:-1]
            ptsC = segment[2:]
            distsA = np.linalg.norm(ptsB - ptsA, axis=-1)
            distsB = np.linalg.norm(ptsC - ptsB, axis=-1)
            distsC = np.linalg.norm(ptsA - ptsC, axis=-1)
            areaABC = np.abs(0.5 * (ptsA[:, 0] * (ptsB[:, 1] - ptsC[:, 1]) + ptsB[:, 0] * (ptsC[:, 1] - ptsA[:, 1]) + ptsC[:, 0] * (ptsA[:, 1] - ptsB[:, 1])))
            K = 4 * areaABC / (distsA * distsB * distsC)
            seg_feature.append(np.sum(K))

            #Mean angular difference
            BA = ptsA - ptsB
            BC = ptsC - ptsB
            cosine = np.einsum('ij,ij->i', BA, BC)
            mean_ang_dif = np.mean(np.arccos(cosine / (np.linalg.norm(BA, axis=-1) * np.linalg.norm(BC, axis=-1))))
            seg_feature.append(mean_ang_dif)

            # #Meam speed
            reg = 1e-3
            delta_segment = next_segments[idx] - curr_segments[idx]
            mean_speed = np.mean(delta_segment / (next_odom - odom + reg))
            seg_feature.append(mean_speed)

            # Apeend label
            seg_feature.append(label)

            scan_feature.append(seg_feature)

        return scan_feature

class BoostedFeatureDetector:
    def __init__(self):
        pass

    def adaboost(self, X, Y, K, nSamples):

        # Adaboost with decision stump classifier as weak classifier
        #
        # INPUT:
        # X         : training examples (numSamples x numDim)
        # Y         : training lables (numSamples x 1)
        # K         : number of weak classifiers to select (scalar)
        #             (the _maximal_ iteration count - possibly abort earlier
        #              when error is zero)
        # nSamples  : number of training examples which are selected in each round (scalar)
        #             The sampling needs to be weighted!
        #             Hint - look at the function 'choice' in package numpy.random
        #
        # OUTPUT:
        # alphaK 	: voting weights (K x 1) - for each round
        # para		: parameters of simple classifier (K x 2) - for each round
        #           : dimension 1 is j
        #           : dimension 2 is theta

        N, _ = X.shape  # total number of training samples

        # Initialize the classifier models
        j = np.zeros(K)
        theta = np.zeros(K)

        alpha = np.zeros(K)  # voting weight for each classifier
        # w = (np.ones(N) / N).reshape(N, 1)  # uniform initialization of sample-weights
        w = np.ones((N, 1))
        w[Y == 1.0] = 1 / np.sum(Y == 1.0) / 2
        w[Y == -1.0] = 1 / np.sum(Y == -1.0) / 2
        w = w / np.sum(w)

        for k in range(K):  # Iterate over all classifiers

            # Sample data with weights
            index = choice(N, nSamples, True, w.ravel())

            X_sampled = X[index, :]
            Y_sampled = Y[index]

            # Train the weak classifier C_k
            j[k], theta[k] = self.simple_classifier(X_sampled, Y_sampled)

            cY = (np.ones(N) * (-1)).reshape(N, 1)  # placeholder for class predictions
            cY[X[:, int(j[k] - 1)] > theta[k]] = 1  # classify

            # Calculate weighted error for given classifier
            temp = np.where([Y[i] != cY[i] for i in range(N)], 1, 0).reshape(N, 1)
            ek = np.sum(w * temp)

            # If the error is zero, the data set is correct classified - break the loop
            if ek < 1.0e-01:
                alpha[k] = 1
                break

            # Compute the voting weight for the weak classifier alpha_k
            alpha[k] = 0.5 * np.log((1 - ek) / ek)

            # Update the weights
            w = w * np.exp((-alpha[k] * (Y * cY)))
            w = w / sum(w)

        alphaK = alpha
        para = np.stack((j, theta), axis=1)

        return alphaK, para

    def simple_classifier(self, X, Y):
        # Select a simple classifier
        #
        # INPUT:
        # X         : training examples (numSamples x numDim)
        # Y         : training lables (numSamples x 1)
        #
        # OUTPUT:
        # theta 	: threshold value for the decision (scalar)
        # j 		: the dimension to "look at" (scalar)

        N, D = X.shape

        # Initialize least error
        le = 1
        j = 1  # dimension
        theta = 0  # decision value

        # Iterate over dimensions, which j to choose
        for jj in range(D):

            # Find interval to choose theta
            val = X[:, jj]  # shape: (100 x 1)

            sVal = np.sort(val)  # TODO: returns unique sorted values, shape: (100 x 1)
            idx = np.argsort(val)
            change = np.where(np.roll(Y[idx], -1) + Y[idx] == 0)[0]  # shape: (36 x 1)

            # Calculate thresholds for which we want to check classifier error.
            # Candidates for theta are always between two points of different classes.

            th = (sVal[change[change < len(X) - 1]] + sVal[change[change < len(X) - 1] + 1]) / 2  # shape: (35 x 1)

            error = np.zeros(len(th))  # error-placeholder for each value of threshold th

            # Iterate over canidates forfor theta
            for t in range(len(th)):
                # Initialize temporary labels for given j and theta
                cY = np.ones(N) * (-1)  # shape: (100 x 1)

                # Classify
                cY[X[:, jj] > th[t]] = 1  # Set all values to one, which are bigger then current threshold

                # Calculate error for given classifier
                error[t] = sum([Y[i] != cY[i] for i in range(N)])

                # Visualize potential threshold values
                # print('J = {0} \t Theta = {1} \t Error = {2}\n'.format(jj, th[t], error[t]))

            le1 = min(error / N)
            ind1 = np.argmin(error / N)
            le2 = min(1 - error / N)
            ind2 = np.argmin(1 - error / N)
            le0 = min([le1, le2, le])

            if le == le0:
                continue
            else:
                le = le0
                j = jj + 1  # Set theta to current value of threshold
                # Choose theta and parity for minimum error
                if le1 == le:
                    theta = th[ind1]
                else:
                    theta = th[ind2]

        return j, theta

    def eval(self, X, alphaK, para):
        # INPUT:
        # para	: parameters of simple classifier (K x 2) - for each round
        #           : dimension 1 is j
        #           : dimension 2 is theta
        # K         : number of classifiers used
        # X         : test data points (numSamples x numDim)
        #
        # OUTPUT:
        # classLabels: labels for data points (numSamples x 1)
        # result     : weighted sum of all the K classifier (numSamples x 1)

        K = para.shape[0]  # number of classifiers
        N = X.shape[0]  # number of test points
        result = np.zeros(N)  # prediction for each test point

        for k in range(K):
            # Initialize temporary labels for given j and theta
            cY = np.ones(N) * (-1)
            # Classify
            cY[X[:, int(para[k, 0] - 1)] > para[k, 1]] = 1

            result += alphaK[k] * cY  # update results with weighted prediction

        classLabels = np.sign(result)  # class-predictions for each test point

        return classLabels, result


parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--cfg", type=str, required=True, help="configuration of the experiment")
args = parser.parse_args()

with open(args.cfg, 'r') as f:
    cfg = yaml.safe_load(f)
    cfg['name'] = os.path.basename(args.cfg).split(".")[0] + cfg['tag']

if __name__ == '__main__':
    print('Preparing data and model')
    split = 'train'
    cls_thresh = 0.5
    dataset = Dataset(split, cfg['dataset'])
    detector = BoostedFeatureDetector()

    kMax = len(dataset.input[0]) * 10  # Number of weak classifiers
    nSamples = 200  # Number of random samples to train each classifier

    # Compute parameters of K classifiers and the voting weight for each classifier
    print('Start training')
    alphaK, para = detector.adaboost(dataset.input, dataset.target.reshape(-1, 1), kMax, nSamples)

    print("Start evaluation")
    #=========test============
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    for idx, data in enumerate(dataset.scans_data):
        # data = [segments, cut_ids, scan, odom_t]
        # segments = [[segment0, label0], ...]
        segments = data[0]
        preds, scores = detector.eval(dataset.scans_feature[idx], alphaK, para)
        segments, preds, scores = nms_predicted_center(segments, preds, scores)
        dets_wp = dataset.dets_wp[idx]

        plt.cla()
        ax.set_aspect('equal')
        ax.set_xlim(-15, 30)
        ax.set_ylim(-5 , 30)

        for i, segment in enumerate(segments):
            if preds[i] > 0 and scores[i] > cls_thresh:
                ax.scatter(segment[..., 0], segment[..., 1], s=3, c='blue')
                c = plt.Circle(np.mean(segment, axis=0), radius=0.5, color='blue', fill=False)
                ax.add_artist(c)
            else:
                ax.scatter(segment[..., 0], segment[..., 1], s=3, c='black')

            for wp in dets_wp:
                c = plt.Circle(wp, radius=0.5, color='r', fill=False)
                ax.add_artist(c)

        # ax.quiver(scan_xy[..., 0], scan_xy[..., 1], target_flow[..., 0], target_flow[..., 1])
        plt.savefig('./../../tmp_imgs/frame_%04d.png' % idx)
        # plt.show()

        # print()

        #========================