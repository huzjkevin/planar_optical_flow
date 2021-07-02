import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
# import utils as u
from . import utils as u
import colorsys

def plot_sequence(X, Y, path, odoms_phi=None, color=None, arrow=None):
    # if color is not None:
    #     plt.style.use('dark_background')
    # plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 6))
    canvas = FigureCanvas(fig)
    w, h = canvas.get_width_height()
    ax = fig.add_subplot(111)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w, h))

    a = []

    for idx, (x, y) in enumerate(zip(X, Y)):
        ax.cla()

        ax.set_aspect('equal')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-5, 15)

        ax.set_title('Frame: {}'. format(idx))
        ax.axis("off")

        if odoms_phi is not None:
            x, y = u.rphi_to_xy(x, y + 2.0 * np.pi + odoms_phi[idx])
        else:
            x, y = u.rphi_to_xy(x, y)

        if color is not None:
            ax.scatter(x, y, s=1, c=color[idx])
            a.append(color[idx][..., 1])
        elif arrow is not None:
            # ax.quiver(x[::5], y[::5], arrow[idx][::5, 0], arrow[idx][::5, 1])
            ax.quiver(x, y, arrow[idx][..., 0], arrow[idx][..., 1])
            ax.quiver([0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [0.0, 2.0], color=[[1.0 ,0.0, 0.0], [0.0, 0.0, 1.0]])
        else:
            ax.scatter(x, y, s=1, c='blue')

        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, -1)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    plt.close(fig)
    writer.release()

# plot flow in hsv under fixed frame direction
def plot_flow_fixed_pose(scans_r, scans_phi, odoms_phi, path, is_xy=False, pred=None, target=None, epe_list=None, aae_list=None):
    gs = GridSpec(1, 3, height_ratios=[1.0], width_ratios=[0.4, 0.4, 0.2])
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax_cw = fig.add_subplot(gs[2])
    canvas = FigureCanvas(fig)
    w, h = canvas.get_width_height()
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w, h))

    # Plot color wheel
    color_wheel(ax_cw)

    # Plot flow
    for idx, (r, phi) in enumerate(zip(scans_r, scans_phi)):
        fig.suptitle('Frame: {}'.format(idx))

        ax1.cla()
        ax2.cla()

        ax1.set_aspect('equal')
        ax1.set_xlim(-25, 25)
        ax1.set_ylim(-15, 25)

        if epe_list is not None and aae_list is not None:
            ax1.set_title(
                'Prediction EPE. error: {:0.3f} m /  AAE. error: {:0.3f} deg'.format(epe_list[idx], aae_list[idx]))
        elif epe_list is not None:
            ax1.set_title('Prediction EPE. error: {:0.3f} m'.format(epe_list[idx]))
        elif aae_list is not None:
            ax1.set_title('Prediction AAE. error: {:0.3f} m'.format(aae_list[idx]))
        else:
            ax1.set_title('Prediction')
        ax1.axis("off")

        ax2.set_aspect('equal')
        ax2.set_xlim(-25, 25)
        ax2.set_ylim(-15, 25)

        ax2.set_title('Target')
        ax2.axis("off")

        if is_xy:
            r, phi = u.xy_to_rphi(r, phi)
            phi += odoms_phi[idx]

        x, y = u.rphi_to_xy(r, phi + odoms_phi[idx])

        if pred is not None:
            ax1.scatter(x, y, s=3, c=pred[idx])
        if target is not None:
            ax2.scatter(x, y, s=3, c=target[idx])
        else:
            ax1.scatter(x, y, s=1, c='blue')
            ax2.scatter(x, y, s=1, c='blue')

        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, -1)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    plt.close(fig)
    writer.release()

from matplotlib.ticker import PercentFormatter
def plot_sequence_arrow_color(X, Y, path, color=None, arrow=None):
    # plt.style.use('dark_background')
    fig = plt.figure(figsize=(30, 16))
    canvas = FigureCanvas(fig)
    w, h = canvas.get_width_height()
    ax = fig.add_subplot(131)
    ax_hist_hue = fig.add_subplot(132)
    ax_hist_val = fig.add_subplot(133)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w, h))

    # writer_hist = cv2.VideoWriter('./../../output/tmp_videos/flow_hist.avi', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w, h))

    a = []

    for idx, (x, y) in enumerate(zip(X, Y)):
        ax.cla()
        ax_hist_hue.cla()
        ax_hist_val.cla()

        ax.set_aspect('equal')
        ax.set_xlim(-25, 25)
        ax.set_ylim(-15, 25)

        ax.set_title('Frame: {}'.format(idx))
        ax.axis("off")

        x, y = u.rphi_to_xy(x, y)

        if color is not None:
            ax.scatter(x, y, s=1, c=color[idx])
            a.append(color[idx][..., 1])
        if arrow is not None:
            # ax.quiver(x[::5], y[::5], arrow[idx][::5, 0], arrow[idx][::5, 1])
            ax.quiver(x, y, arrow[idx][..., 0], arrow[idx][..., 1])


        num_bins = 50

        # N is the count in each bin, bins is the lower-limit of the bin
        hue = color[idx][..., 0]
        val = color[idx][..., 2]
        N1, bins1, patches1 = ax_hist_hue.hist(hue, bins=num_bins, weights=np.ones(len(hue)) / len(hue))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax_hist_hue.set_title("histogram-hue")
        ax_hist_hue.set_xlabel('hue')

        N2, bins2, patches2 = ax_hist_val.hist(val, bins=num_bins, weights=np.ones(len(val)) / len(val))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax_hist_val.set_title("histogram-val")
        ax_hist_val.set_xlabel('val')

        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, -1)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    plt.close(fig)
    writer.release()

def plot_sequence_xy(X, Y, path, color=None, arrow=None):
    if color is not None:
        plt.style.use('dark_background')
    # plt.style.use('dark_background')
    fig = plt.figure(figsize=(5, 4))
    canvas = FigureCanvas(fig)
    w, h = canvas.get_width_height()
    ax = fig.add_subplot(111)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w, h))

    a = []

    for idx, (x, y) in enumerate(zip(X, Y)):
        plt.cla()

        ax.set_aspect('equal')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-5, 15)

        ax.set_title('Frame: {}'. format(idx))
        ax.axis("off")

        # x, y = u.rphi_to_xy(x, y)

        if color is not None:
            ax.scatter(x, y, s=1, c=color[idx])
            a.append(color[idx][..., 1])
        elif arrow is not None:
            ax.quiver(x[::5], y[::5], arrow[idx][::5, 0], arrow[idx][::5, 1])
            # ax.quiver(x, y, arrow[idx][..., 0], arrow[idx][..., 1])
        else:
            ax.scatter(x, y, s=1, c='blue')

        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, -1)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    plt.close(fig)

def plot_sequence_xy_test(X, Y, path, masks=None, color=None, arrow=None):
    if color is not None:
        plt.style.use('dark_background')
    # plt.style.use('dark_background')
    fig = plt.figure(figsize=(5, 4))
    canvas = FigureCanvas(fig)
    w, h = canvas.get_width_height()
    ax = fig.add_subplot(111)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w, h))

    a = []

    for idx, (x, y, mask) in enumerate(zip(X, Y, masks)):
        plt.cla()

        ax.set_aspect('equal')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-5, 15)

        ax.set_title('Frame: {}'. format(idx))
        ax.axis("off")

        # a = x[mask.reshape(-1) == 1.0]
        x_static = x[mask.reshape(-1) == 1.0]
        y_static = y[mask.reshape(-1) == 1.0]
        ax.scatter(x_static, y_static, s=3, c='blue')

        x_dynamic = x[mask.reshape(-1) == 0.0]
        y_dynamic = y[mask.reshape(-1) == 0.0]
        ax.scatter(x_dynamic, y_dynamic, s=3, c='red')

        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, -1)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    plt.close(fig)

def plot_sequence_gt_pred_arrow(X, Y, path, pred=None, target=None, epe_list=None, aae_list=None):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(24, 16)
    canvas = FigureCanvas(fig)
    w, h = canvas.get_width_height()
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w, h))

    for idx, (x, y) in enumerate(zip(X, Y)):
        fig.suptitle('Frame: {}'.format(idx))

        ax1.cla()
        ax2.cla()

        ax1.set_aspect('equal')
        ax1.set_xlim(-15, 15)
        ax1.set_ylim(-5, 15)

        if epe_list is not None and aae_list is not None:
            ax1.set_title(
                'Prediction EPE. error: {:0.3f} m /  AAE. error: {:0.3f} deg'.format(epe_list[idx], aae_list[idx]))
        elif epe_list is not None:
            ax1.set_title('Prediction EPE. error: {:0.3f} m'.format(epe_list[idx]))
        elif aae_list is not None:
            ax1.set_title('Prediction AAE. error: {:0.3f} m'.format(aae_list[idx]))
        else:
            ax1.set_title('Prediction')
        ax1.axis("off")

        ax2.set_aspect('equal')
        ax2.set_xlim(-15, 15)
        ax2.set_ylim(-5, 15)

        ax2.set_title('Target')
        ax2.axis("off")

        # x, y = u.rphi_to_xy(x, y)

        if pred is not None:
            # ax1.quiver(x[::5], y[::5], pred[idx][::5, 0], pred[idx][::5, 1])
            ax1.quiver(x, y, pred[idx][..., 0], pred[idx][..., 1], units='xy', scale_units='xy', scale=1)
        if target is not None:
            # ax2.quiver(x[::5], y[::5], target[idx][::5, 0], target[idx][::5, 1])
            ax2.quiver(x, y, target[idx][..., 0], target[idx][..., 1], units='xy', scale_units='xy', scale=1)
            # ax.quiver(x, y, arrow[idx][..., 0], arrow[idx][..., 1])
        else:
            ax1.scatter(x, y, s=1, c='blue')
            ax2.scatter(x, y, s=1, c='blue')

        # plt.savefig('./tmp_imgs/frame_%04d.png' % idx)

        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, -1)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    plt.close(fig)

def plot_sequence_gt_pred_hsv(X, Y, path, pred=None, target=None, epe_list=None, aae_list=None):
    gs = GridSpec(1, 3, height_ratios=[1.0], width_ratios=[0.4, 0.4, 0.2])
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax_cw = fig.add_subplot(gs[2])
    canvas = FigureCanvas(fig)
    w, h = canvas.get_width_height()
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w, h))

    # Plot color wheel
    color_wheel(ax_cw)

    # Plot flow
    for idx, (x, y) in enumerate(zip(X, Y)):
        fig.suptitle('Frame: {}'.format(idx))

        ax1.cla()
        ax2.cla()

        ax1.set_aspect('equal')
        ax1.set_xlim(-25, 25)
        ax1.set_ylim(-15, 25)

        if epe_list is not None and aae_list is not None:
            ax1.set_title('Prediction EPE. error: {:0.3f} m /  AAE. error: {:0.3f} deg'.format(epe_list[idx], aae_list[idx]))
        elif epe_list is not None:
            ax1.set_title('Prediction EPE. error: {:0.3f} m'.format(epe_list[idx]))
        elif aae_list is not None:
            ax1.set_title('Prediction AAE. error: {:0.3f} m'.format(aae_list[idx]))
        else:
            ax1.set_title('Prediction')
        ax1.axis("off")

        ax2.set_aspect('equal')
        ax2.set_xlim(-25, 25)
        ax2.set_ylim(-15, 25)

        ax2.set_title('Target')
        ax2.axis("off")

        # x, y = u.rphi_to_xy(x, y)

        if pred is not None:
            ax1.scatter(x, y, s=3, c=pred[idx])
        if target is not None:
            ax2.scatter(x, y, s=3, c=target[idx])
        else:
            ax1.scatter(x, y, s=1, c='blue')
            ax2.scatter(x, y, s=1, c='blue')

        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, -1)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    plt.close(fig)

def play_sequence_test():
    # scans
    seq_name = './../../data/DROWv2-data/train/run_2015-11-25-10-34-20-b.bag.csv'
    scans_data = np.genfromtxt(seq_name, delimiter=',')
    scans_t = scans_data[:, 1]
    scans = scans_data[:, 2:]
    scan_phi = u.get_laser_phi()

    # odometry, used only for plotting
    odo_name = seq_name[:-3] + 'odom2'
    odos = np.genfromtxt(odo_name, delimiter=',')
    odos_t = odos[:, 1]
    odos_phi = odos[:, 4]

    # plot
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    # video sequence
    odo_idx = 0
    for i in range(len(scans[:100])):
        plt.cla()

        ax.set_aspect('equal')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-5, 15)

        # ax.set_title('Frame: %s' % i)
        ax.set_title('Press any key to exit.')
        ax.axis("off")

        # plot points
        scan = scans[i]
        scan_x, scan_y = u.rphi_to_xy(scan, scan_phi)
        ax.scatter(scan_x, scan_y, s=1, c='blue')

        plt.savefig('./../../tmp_imgs/frame_%04d.png' % i)

def color_wheel(ax, resolution=200, radius=3.0):
    # ax.set_title('Color wheel')
    ax.set_aspect('equal')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.axis("off")

    xy_list, color_list = [], []
    r_grid = np.linspace(0.0, radius, resolution)
    phi_grid = np.linspace(0, 2.0 * np.pi, resolution)

    for phi in phi_grid:
        # print(phi)
        for r in r_grid:
            x, y = u.rphi_to_xy(r, phi)
            hsv = np.array([((phi + 2.0 * np.pi) / np.pi / 2),
                            (r / 5.0),
                            1.0])

            color = np.array(colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2]))
            xy_list.append([x, y])
            color_list.append(color)

    xy_list = np.array(xy_list)
    color_list = np.array(color_list)

    ax.scatter(xy_list[..., 0], xy_list[..., 1], s=5, c=color_list)

    return ax

def plot_person_flow(scans, dets_xy, dets_cls, instance_masks, path, is_xy=False, pred=None, target=None, epe_list=None, aae_list=None):
    gs = GridSpec(1, 2, height_ratios=[1.0], width_ratios=[0.8, 0.2])
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(gs[0])
    # ax2 = fig.add_subplot(132)
    ax_cw = fig.add_subplot(gs[1])
    canvas = FigureCanvas(fig)
    w, h = canvas.get_width_height()
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w, h))

    # Plot color wheel
    color_wheel(ax_cw)

    # Plot flow
    for idx, (det_xy, det_cls, instance_mask) in enumerate(zip(dets_xy, dets_cls, instance_masks)):
        fig.suptitle('Frame: {}'.format(idx))

        ax1.cla()
        # ax2.cla()

        ax1.set_aspect('equal')
        ax1.set_xlim(-15, 15)
        ax1.set_ylim(-5, 15)

        if epe_list is not None and aae_list is not None:
            ax1.set_title(
                'Prediction EPE. error: {:0.3f} m /  AAE. error: {:0.3f} deg'.format(epe_list[idx], aae_list[idx]))
        elif epe_list is not None:
            ax1.set_title('Prediction EPE. error: {:0.3f} m'.format(epe_list[idx]))
        elif aae_list is not None:
            ax1.set_title('Prediction AAE. error: {:0.3f} m'.format(aae_list[idx]))
        else:
            ax1.set_title('Prediction')
        ax1.axis("off")

        # ax2.set_aspect('equal')
        # ax2.set_xlim(-25, 25)
        # ax2.set_ylim(-15, 25)
        #
        # ax2.set_title('Target')
        # ax2.axis("off")

        phi_grid = u.get_laser_phi()
        x, y = u.rphi_to_xy(scans[idx], phi_grid)

        for i in range(len(det_cls)):
            if det_cls[i][0] < 0.3:
                instance_ids = instance_mask == i + 1
                ax1.scatter(x[instance_ids], y[instance_ids], s=3, c='black')
            else:
                instance_ids = instance_mask == i + 1
                ax1.scatter(x[instance_ids], y[instance_ids], s=3, c=pred[idx][instance_ids])

                c = plt.Circle(det_xy[i], radius=0.5, color='r', fill=False)
                # ax1.text(det_xy[i][0], det_xy[i][1], s=i, fontsize=12)
                ax1.add_artist(c)

        # ax2.scatter(x, y, s=3, c=target[idx])

        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, -1)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    plt.close(fig)
    writer.release()

def plot_person_flow_fixed_pose(scans, dets_xy, dets_cls, instance_masks, path, odoms_phi, pred_hsv, pred_arrow, det_thresh=0.3):
    gs = GridSpec(1, 2, height_ratios=[1.0], width_ratios=[0.9, 0.1])
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(gs[0])
    # ax2 = fig.add_subplot(132)
    ax_cw = fig.add_subplot(gs[1])
    canvas = FigureCanvas(fig)
    w, h = canvas.get_width_height()
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w, h))

    # Plot color wheel
    # color_wheel(ax_cw)
    phi_grid = u.get_laser_phi()
    # scanner location
    rad_tmp = 0.5 * np.ones(len(phi_grid), dtype=np.float)
    xy_scanner = u.rphi_to_xy(rad_tmp, phi_grid)
    xy_scanner = np.stack(xy_scanner, axis=1)

    # Plot flow
    for idx, (det_xy, det_cls, instance_mask) in enumerate(zip(dets_xy, dets_cls, instance_masks)):
        fig.suptitle('Frame: {}'.format(idx))

        ax_cw.axis('off')

        ax1.cla()

        ax1.set_aspect('equal')
        ax1.set_xlim(-15, 15)
        ax1.set_ylim(-5, 15)

        ax1.axis("off")

        x, y = u.rphi_to_xy(scans[idx], phi_grid + odoms_phi[idx])
        # x, y = u.rphi_to_xy(scans[idx], phi_grid)

        rot = np.array([[np.cos(odoms_phi[idx]), np.sin(odoms_phi[idx])],
                        [-np.sin(odoms_phi[idx]), np.cos(odoms_phi[idx])]], dtype=np.float32)

        # plot scanner location
        xy_scanner_rot = np.matmul(xy_scanner, rot.T)
        ax1.plot(xy_scanner_rot[:, 0], xy_scanner_rot[:, 1], c='black')
        ax1.plot((0, xy_scanner_rot[0, 0] * 1.0), (0, xy_scanner_rot[0, 1] * 1.0), c='black')
        ax1.plot((0, xy_scanner_rot[-1, 0] * 1.0), (0, xy_scanner_rot[-1, 1] * 1.0), c='black')

        for i in range(len(det_cls)):
            if det_cls[i][0] < det_thresh:
                instance_ids = instance_mask == i + 1
                ax1.scatter(x[instance_ids], y[instance_ids], s=3, c='black')
            else:
                instance_ids = instance_mask == i + 1
                det_xy_rot = np.matmul(det_xy[i], rot.T)
                if pred_arrow is not None:
                    ax1.quiver(det_xy_rot[0], det_xy_rot[1],
                               np.mean(pred_arrow[idx][instance_ids, 0]),
                               np.mean(pred_arrow[idx][instance_ids, 1]),
                               color=np.mean(pred_hsv[idx][instance_ids], axis=0),
                               linewidth=0.1)
                    ax1.scatter(x[instance_ids], y[instance_ids], s=3, c=pred_hsv[idx][instance_ids])
                else:
                    ax1.scatter(x[instance_ids], y[instance_ids], s=3, c=pred_hsv[idx][instance_ids])

                c = plt.Circle(det_xy_rot, radius=0.5, color='r', fill=False)
                # ax1.text(det_xy[i][0], det_xy[i][1], s=i, fontsize=12)
                ax1.add_artist(c)

        # ax1.quiver(x, y, pred_arrow[idx][..., 0], pred_arrow[idx][..., 1], linewidth=0.1)

        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, -1)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    plt.close(fig)
    writer.release()

if __name__ == "__main__":
    X = np.linspace(0,15,100)
    Y = np.ones_like(X)
    flow = np.zeros((100, 2))
    flow[..., 0] = 0.15
    colors = u.flow_to_hsv(flow)

    path = "./../../tmp_videos/dump_sample.avi"

    plot_sequence(X, Y, path=path)
    print("finished")



