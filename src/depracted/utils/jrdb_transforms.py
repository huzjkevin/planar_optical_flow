import numpy as np

"""
Transformations. Following frames are defined:

base: main frame where 3D annotations are done in, x-forward, y-left, z-up
upper_lidar: x-forward, y-left, z-up
lower_lidar: x-forward, y-left, z-up
laser: x-forward, y-left, z-up
"""


def _get_R_z(rot_z):
    cs, ss = np.cos(rot_z), np.sin(rot_z)
    return np.array([[cs, -ss, 0], [ss, cs, 0], [0, 0, 1]], dtype=np.float32)


# laser to base
_rot_z_laser_to_base = np.pi / 120
_R_laser_to_base = _get_R_z(_rot_z_laser_to_base)

# upper_lidar to base
_rot_z_upper_lidar_to_base = 0.085
_T_upper_lidar_to_base = np.array([0, 0, 0.33529], dtype=np.float32).reshape(3, 1)
_R_upper_lidar_to_base = _get_R_z(_rot_z_upper_lidar_to_base)

# lower_lidar to base
_rot_z_lower_lidar_to_base = 0.0
_T_lower_lidar_to_base = np.array([0, 0, -0.13511], dtype=np.float32).reshape(3, 1)
_R_lower_lidar_to_base = np.eye(3, dtype=np.float32)


"""
Transformation API
"""


def transform_pts_upper_velodyne_to_base(pts):
    """Transform points from upper velodyne frame to base frame

    Args:
        pts (np.array[3, N]): points (x, y, z)

    Returns:
        pts_trans (np.array[3, N])
    """
    return _R_upper_lidar_to_base @ pts + _T_upper_lidar_to_base


def transform_pts_lower_velodyne_to_base(pts):
    return _R_lower_lidar_to_base @ pts + _T_lower_lidar_to_base


def transform_pts_laser_to_base(pts):
    return _R_laser_to_base @ pts


def transform_pts_base_to_upper_velodyne(pts):
    return _R_upper_lidar_to_base.T @ (pts - _T_upper_lidar_to_base)


def transform_pts_base_to_lower_velodyne(pts):
    return _R_lower_lidar_to_base.T @ (pts - _T_lower_lidar_to_base)


def transform_pts_base_to_laser(pts):
    return _R_laser_to_base.T @ pts


def box_from_jrdb(jrdb_label):
    xyz = np.array(
        [jrdb_label["cx"], jrdb_label["cy"], jrdb_label["cz"]], dtype=np.float32
    )
    lwh = np.array(
        [jrdb_label["l"], jrdb_label["w"], jrdb_label["h"]], dtype=np.float32
    )
    rot_z = jrdb_label["rot_z"]

    return Box3d(xyz, lwh, rot_z)


class Box3d:
    def __init__(self, xyz, lwh, rot_z):
        """A 3D bounding box.

        Args:
            xyz (np.array(3,)): center xyz
            lwh (np.array(3,)): length, width, height. When rot_z is zero, length
                should be aligned with x axis
            rot_z (float): rotation around z axis
        """
        self.xyz = xyz.reshape(3, 1)
        self.lwh = lwh.reshape(3, 1)
        self.rot_z = rot_z

    def to_vertices(self):
        """Return xyz coordinates of the eight vertices of the bounding box

        First four points are fl (front left), fr, br, bl on top plane. Last four
        points are same order, but for the bottom plane.

        Returns:
            vert_xyz (np.array([3, 8]))
        """
        vert_xyz = np.array(
            [
                [1, 1, -1, -1, 1, 1, -1, -1],
                [-1, 1, 1, -1, -1, 1, 1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1],
            ],
            dtype=np.float32,
        )
        vert_xyz = 0.5 * vert_xyz * self.lwh

        # NOTE plus pi specifically for JRDB, don't know the reason exactly
        cs, ss = np.cos(self.rot_z + np.pi), np.sin(self.rot_z + np.pi)
        R = np.array([[cs, ss, 0], [-ss, cs, 0], [0, 0, 1]], dtype=np.float32)
        vert_xyz = R @ vert_xyz + self.xyz

        return vert_xyz

    def draw_bev(self, ax, c="red"):
        vert_xyz = self.to_vertices()

        # side and back boarder
        xy = vert_xyz[:2, [1, 2, 3, 0]]
        ax.plot(xy[0], xy[1], c=c, linestyle="-")

        # front boarder
        xy = vert_xyz[:2, [0, 1]]
        ax.plot(xy[0], xy[1], c=c, linestyle="--")

    def draw_fpv(self, ax, dim, c="red"):
        """Plot first person view.

        Args:
            ax: Axes handle of matplotlib
            dim (int): 0 (x) for xz plot or 1 (y) for yz plot
            c (str, optional): color. Defaults to "red".
        """
        vert_xyz = self.to_vertices()

        # top
        x = vert_xyz[dim, [0, 1, 2, 3, 0]]
        z = vert_xyz[2, [0, 1, 2, 3, 0]]
        ax.plot(x, z, c=c, linestyle="-")

        # bottom
        x = vert_xyz[dim, [4, 5, 6, 7, 4]]
        z = vert_xyz[2, [4, 5, 6, 7, 4]]
        ax.plot(x, z, c=c, linestyle="-")

        # vertical bar
        for i in range(4):
            x = vert_xyz[dim, [i, i + 4]]
            z = vert_xyz[2, [i, i + 4]]
            ax.plot(x, z, c=c, linestyle="-")

        # mark orientation
        x = vert_xyz[dim, [0, 5]]
        z = vert_xyz[2, [0, 5]]
        ax.plot(x, z, c=c, linestyle="--")
        x = vert_xyz[dim, [1, 4]]
        z = vert_xyz[2, [1, 4]]
        ax.plot(x, z, c=c, linestyle="--")
