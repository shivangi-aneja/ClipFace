# Borrowed from Pytorch3D

import torch
import numpy as np
from torch import nn


class Materials(nn.Module):
    """
    A class for storing a batch of material properties. Currently only one
    material per batch element is supported.
    """

    def __init__(self, ambient_color=np.array([1., 1., 1.]), diffuse_color=np.array([1., 1., 1.]),
                 specular_color=np.array([1., 1., 1.]), shininess=np.array([32])) -> None:
        """
        Args:
            ambient_color: RGB ambient reflectivity of the material
            diffuse_color: RGB diffuse reflectivity of the material
            specular_color: RGB specular reflectivity of the material
            shininess: The specular exponent for the material. This defines
                the focus of the specular highlight with a high value
                resulting in a concentrated highlight. Shininess values
                can range from 0-1000.

        ambient_color, diffuse_color and specular_color can be of shape
        (1, 3) or (N, 3). shininess can be of shape (1) or (N).

        The colors and shininess are broadcast against each other so need to
        have either the same batch dimension or batch dimension = 1.
        """
        super().__init__()
        self.diffuse_color = torch.from_numpy(diffuse_color)
        self.ambient_color = torch.from_numpy(ambient_color)
        self.specular_color = torch.from_numpy(specular_color)
        self.shininess = torch.from_numpy(shininess)
        _validate_light_properties(self)

    def set_device(self, device):
        self.ambient_color = self.ambient_color.to(device)
        self.diffuse_color = self.diffuse_color.to(device)
        self.specular_color = self.specular_color.to(device)
        self.shininess = self.shininess.to(device)

    def clone(self):
        other = Materials()
        return super().clone(other)


def _validate_light_properties(obj):
    props = ("ambient_color", "diffuse_color", "specular_color")
    for n in props:
        t = getattr(obj, n)
        if t.shape[-1] != 3:
            msg = "Expected %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
