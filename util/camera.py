# borrowed form keunhong/toolbox
import numpy as np
import math
from numpy import linalg


def normalized(vec):
    return vec / linalg.norm(vec)


class BaseCamera:
    def __init__(self, size, near, far, clear_color=(1.0, 1.0, 1.0, 1.0)):
        self.size = size
        self.near = near
        self.far = far
        self.clear_color = clear_color
        if len(self.clear_color) == 3:
            self.clear_color = (*self.clear_color, 1.0)
        self.position = None
        self.up = None
        self.lookat = None

    @property
    def left(self):
        return -self.size[0] / 2

    @property
    def right(self):
        return self.size[0] / 2

    @property
    def top(self):
        return self.size[1] / 2

    @property
    def bottom(self):
        return -self.size[1] / 2

    @property
    def forward(self):
        return normalized(np.subtract(self.lookat, self.position))

    def projection_mat(self):
        raise NotImplementedError

    def rotation_mat(self):
        rotation_mat = np.eye(3)
        rotation_mat[0, :] = normalized(np.cross(self.forward, self.up))
        rotation_mat[2, :] = -self.forward
        # We recompute the 'up' vector portion of the matrix as the cross
        # product of the forward and sideways vector so that we have an ortho-
        # normal basis.
        rotation_mat[1, :] = np.cross(rotation_mat[2, :], rotation_mat[0, :])
        return rotation_mat

    def translation_vec(self):
        rotation_mat = self.rotation_mat()
        return -rotation_mat.T @ self.position

    def view_mat(self):
        rotation_mat = self.rotation_mat()
        position = rotation_mat.dot(self.position)

        view_mat = np.eye(4)
        view_mat[:3, :3] = rotation_mat
        view_mat[:3, 3] = -position

        return view_mat

    def cam_to_world(self):
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = self.rotation_mat().T
        cam_to_world[:3, 3] = self.position
        return cam_to_world

    def handle_mouse(self, last_pos, cur_pos):
        pass

    def apply_projection(self, points):
        homo = euclidean_to_homogeneous(points)
        proj = self.projection_mat().dot(self.view_mat().dot(homo.T)).T
        proj = homogeneous_to_euclidean(proj)[:, :2]
        proj = (proj + 1) / 2
        proj[:, 0] = (proj[:, 0] * self.size[0])
        proj[:, 1] = self.size[1] - (proj[:, 1] * self.size[1])
        return np.fliplr(proj)

    def get_position(self):
        return linalg.inv(self.view_mat())[:3, 3]

    def serialize(self):
        raise NotImplementedError()


class PerspectiveCamera(BaseCamera):

    def __init__(self, size, near, far, fov, position, lookat, up,
                 *args, **kwargs):
        super().__init__(size, near, far, *args, **kwargs)

        self.fov = fov
        self._position = np.array(position, dtype=np.float32)
        self.lookat = np.array(lookat, dtype=np.float32)
        self.up = normalized(np.array(up))

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = np.array(position)

    def projection_mat(self):
        mat = perspective(self.fov, self.size[0] / self.size[1], self.near, self.far).T
        return mat

    def view_mat(self):
        rotation_mat = np.eye(3)
        rotation_mat[0, :] = normalized(np.cross(self.forward, self.up))
        rotation_mat[2, :] = -self.forward
        # We recompute the 'up' vector portion of the matrix as the cross
        # product of the forward and sideways vector so that we have an ortho-
        # normal basis.
        rotation_mat[1, :] = np.cross(rotation_mat[2, :], rotation_mat[0, :])

        position = rotation_mat.dot(self.position)

        view_mat = np.eye(4)
        view_mat[:3, :3] = rotation_mat
        view_mat[:3, 3] = -position
        return view_mat

    def serialize(self):
        return {
            'type': 'perspective',
            'size': self.size,
            'near': float(self.near),
            'far': float(self.far),
            'fov': self.fov,
            'position': self.position.tolist(),
            'lookat': self.lookat.tolist(),
            'up': self.up.tolist(),
            'clear_color': self.clear_color,
        }


class OrthographicCamera(BaseCamera):

    def __init__(self, size, near, far, position, lookat, up, *args, **kwargs):
        super().__init__(size, near, far, *args, **kwargs)
        self.lookat = lookat
        self.position = position
        self.up = up

    def projection_mat(self):
        assert (self.near != self.far)
        aspect = self.size[0] / self.size[1]
        h = self.near
        w = h * aspect
        mat = orthographic_projection(left=-w, right=w, bottom=h, top=-h, znear=self.near, zfar=self.far).T
        return mat

    def view_mat(self):
        rotation_mat = np.eye(3)
        rotation_mat[0, :] = normalized(np.cross(self.forward, self.up))
        rotation_mat[2, :] = -self.forward
        # We recompute the 'up' vector portion of the matrix as the cross
        # product of the forward and sideways vector so that we have an ortho-
        # normal basis.
        rotation_mat[1, :] = np.cross(rotation_mat[2, :], rotation_mat[0, :])

        position = rotation_mat.dot(self.position)

        view_mat = np.eye(4)
        view_mat[:3, :3] = rotation_mat
        view_mat[:3, 3] = -position
        return view_mat

    def serialize(self):
        return {
            'type': 'orthographic',
            'size': self.size,
            'near': float(self.near),
            'far': float(self.far),
            'position': self.position.tolist(),
            'lookat': self.lookat.tolist(),
            'up': self.up.tolist(),
            'clear_color': self.clear_color,
        }


def spherical_to_cartesian(radius, azimuth, elevation):
    x = radius * math.cos(azimuth + 3 * math.pi / 2) * math.sin(elevation)
    y = radius * math.cos(elevation)
    z = radius * math.sin(azimuth + 3 * math.pi / 2) * math.sin(elevation)
    return x, y, z


def spherical_coord_to_cam(fov, azimuth, elevation, max_len=500, cam_dist=1.75, lookat=(0, 0, 0)):
    shape = (max_len * 2, max_len * 2)
    camera = PerspectiveCamera(
        size=shape, fov=fov, near=0.1, far=5000.0,
        position=(0, 0, -cam_dist), clear_color=(1, 1, 1, 1),
        lookat=lookat, up=(0, 1, 0))
    camera.position = spherical_to_cartesian(cam_dist, azimuth, elevation)
    return camera


def euclidean_to_homogeneous(points):
    ones = np.ones((points.shape[0], 1))
    return np.concatenate((points, ones), 1)


def homogeneous_to_euclidean(points):
    ndims = points.shape[1]
    euclidean_points = np.array(points[:, 0:ndims - 1]) / points[:, -1, None]
    return euclidean_points


def perspective(fovy, aspect, znear, zfar):
    assert(znear != zfar)
    h = math.tan(fovy / 360.0 * math.pi) * znear
    w = h * aspect
    return frustum_perspective(-w, w, h, -h, znear, zfar)


def frustum_perspective(left, right, bottom, top, znear, zfar):
    """
        From: https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
    """
    # From Scratchpixel, uses row vectors
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / float(right - left)
    M[2, 0] = (right + left) / float(right - left)
    M[1, 1] = +2.0 * znear / float(top - bottom)
    M[2, 1] = (top + bottom) / float(top - bottom)
    M[2, 2] = -(zfar + znear) / float(zfar - znear)
    M[3, 2] = -2.0 * znear * zfar / float(zfar - znear)
    M[2, 3] = -1.0
    return M


def orthographic_projection(left, right, bottom, top, znear, zfar):
    """
        From: https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/orthographic-projection-matrix
    """
    # From Scratchpixel, uses row vectors
    assert (right != left)
    assert (bottom != top)
    assert (znear != zfar)

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 / float(right - left)
    M[3, 0] = - (right + left) / float(right - left)
    M[1, 1] = + 2.0 / float(top - bottom)
    M[3, 1] = - (top + bottom) / float(top - bottom)
    M[2, 2] = -2.0 / float(zfar - znear)
    M[3, 2] = - (zfar + znear) / float(zfar - znear)
    M[3, 3] = 1.0
    return M
