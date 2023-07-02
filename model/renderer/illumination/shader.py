# Borrowed from Pytorch3D

import torch
import torch.nn as nn
from .utils import BlendParams
from .lighting import PointLight
from .materials import Materials
from .shading import apply_lighting


class SoftPhongShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::
        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(self, lights=None, materials=None) -> None:
        super().__init__()
        self.lights = lights if lights is not None else PointLight()
        self.materials = (materials if materials is not None else Materials())

    def set_device(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.lights.set_device(device)
        self.materials.set_device(device)

    def forward(self, points, normals, camera_position):

        colors = apply_lighting(points=points, normals=normals, lights=self.lights, camera_position=camera_position,
                                materials=self.materials)

        return colors
