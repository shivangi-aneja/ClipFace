# Borrowed from Pytorch3D

import torch
from typing import Tuple


def normalize_in_range(color):
    color_255 = color * 255.0
    color_norm = color_255 / 127.5 - 1
    return color_norm


def apply_lighting(points, normals, lights, camera_position, materials) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        points: torch tensor of shape (N, ..., 3) or (P, 3).
        normals: torch tensor of shape (N, ..., 3) or (P, 3)
        lights: instance of the Lights class.
        camera_position: Camera Position.
        materials: instance of the Materials class.

    Returns:
        ambient_color: same shape as materials.ambient_color
        diffuse_color: same shape as the input points
        specular_color: same shape as the input points
    """
    light_diffuse = lights.diffuse(normals=normals, points=points)
    light_specular = lights.specular(normals=normals, points=points, camera_position=camera_position,
                                     shininess=materials.shininess, )
    ambient_color = normalize_in_range(materials.ambient_color * lights.ambient_color)
    diffuse_color = normalize_in_range(materials.diffuse_color * light_diffuse)
    specular_color = normalize_in_range(materials.specular_color * light_specular)
    # diffuse_color which is of shape (N, H, W, 3)
    ambient_color = ambient_color[:, None, None, :]
    return ambient_color, diffuse_color, specular_color
