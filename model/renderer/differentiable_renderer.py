import torch
from torch import nn
import nvdiffrast.torch as dr
from .illumination.shader import SoftPhongShader
from torchvision.ops import masks_to_boxes


def transform_pos(pos, projection_matrix, world_to_cam_matrix):
    # (x,y,z) -> (x,y,z,1)
    t_mtx = torch.matmul(projection_matrix, world_to_cam_matrix)
    # noinspection PyArgumentList
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
    return torch.matmul(posw, t_mtx.t())


def transform_pos_mvp(pos, mvp):
    """
        Apply projection and view matrices multiplication to the mesh vertices
    Args:
        pos: Given mesh (N, 3)
        mvp: The combined view-projection matrix to be multiplied with

    Returns:

    """
    # noinspection PyArgumentList
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)   # Adding the w-coordinate
    posh = torch.bmm(posw.unsqueeze(0).expand(mvp.shape[0], -1, -1), mvp.permute((0, 2, 1))).reshape((-1, 4))
    pos = torch.zeros_like(posh)
    pos[:, :3] = posh[:, :3] / posh[:, 3:4]
    pos[:, 3] = 1
    return pos


def render_with_texture(glctx, pos_clip, pos_idx, uv_coords, tex_image, col_idx, resolution, ranges, background=None, vertex_normals=None):
    render_resolution = int(resolution)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[render_resolution, render_resolution], ranges=ranges)
    texc, texd = dr.interpolate(uv_coords[None, ...], rast_out, col_idx, rast_db=rast_out_db, diff_attrs='all')
    # color = dr.texture(tex_image[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=9)
    color = dr.texture(tex_image.permute(0, 2, 3, 1).contiguous(), texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=9)
    mask = rast_out[..., -1:] == 0
    if background is None:
        one_tensor = torch.ones((color.shape[0], color.shape[3], 1, 1), device=color.device)
    else:
        one_tensor = background
    one_tensor_permuted = one_tensor.permute((0, 2, 3, 1)).contiguous()
    color = torch.where(mask, one_tensor_permuted, color)  # [:, :, :, :-1]
    color = torch.cat((color, mask.float()), dim=-1)
    return color[:, :, :, :-1]


def render_in_bounds_with_texture(glctx, pos_clip, pos_idx, uv_coords, tex_image, col_idx, resolution, ranges, background=None, vertex_normals=None):
    render_resolution = int(resolution * 1.2)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[render_resolution, render_resolution], ranges=ranges)
    texc, texd = dr.interpolate(uv_coords[None, ...], rast_out, col_idx, rast_db=rast_out_db, diff_attrs='all')
    color = dr.texture(tex_image[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=9)
    mask = rast_out[..., -1:] == 0
    if background is None:
        one_tensor = torch.ones((color.shape[0], color.shape[3], 1, 1), device=color.device)
    else:
        one_tensor = background
    one_tensor_permuted = one_tensor.permute((0, 2, 3, 1)).contiguous()
    color = torch.where(mask, one_tensor_permuted, color)  # [:, :, :, :-1]
    color = torch.cat([color, mask], dim=-1)
    color_crops = []
    boxes = masks_to_boxes(torch.logical_not(mask.squeeze(-1)))
    for img_idx in range(color.shape[0]):
        x1, y1, x2, y2 = [int(val) for val in boxes[img_idx, :].tolist()]
        color_crop = color[img_idx, y1: y2, x1: x2, :].permute((2, 0, 1))
        pad = [[0, 0], [0, 0]]
        if y2 - y1 > x2 - x1:
            total_pad = (y2 - y1) - (x2 - x1)
            pad[0][0] = total_pad // 2
            pad[0][1] = total_pad - pad[0][0]
            pad[1][0], pad[1][1] = 0, 0
            additional_pad = int((y2 - y1) * 0.1)
        else:
            total_pad = (x2 - x1) - (y2 - y1)
            pad[0][0], pad[0][1] = 0, 0
            pad[1][0] = total_pad // 2
            pad[1][1] = total_pad - pad[1][0]
            additional_pad = int((x2 - x1) * 0.1)
        for i in range(4):
            pad[i // 2][i % 2] += additional_pad

        padded = torch.ones((color_crop.shape[0], color_crop.shape[1] + pad[1][0] + pad[1][1], color_crop.shape[2] + pad[0][0] + pad[0][1]), device=color_crop.device)
        padded[:3, :, :] = padded[:3, :, :] * one_tensor[img_idx, :3, :, :]
        padded[:, pad[1][0]: padded.shape[1] - pad[1][1], pad[0][0]: padded.shape[2] - pad[0][1]] = color_crop
        color_crop = torch.nn.functional.interpolate(padded.unsqueeze(0), size=(resolution, resolution), mode='bilinear', align_corners=False).permute((0, 2, 3, 1))
        color_crops.append(color_crop)
    return torch.cat(color_crops, dim=0)


def render_with_texture_shading(glctx, shader, vertex_positions, triface_indices, uv_coords, uv_indices, resolution, ranges,
                                background=None, vertex_normals=None, vertex_positions_world=None):
    render_resolution = int(resolution)
    rast_out, rast_out_db = dr.rasterize(glctx, vertex_positions, triface_indices, resolution=[render_resolution, render_resolution], ranges=ranges)
    mask = rast_out[..., -1:] == 0
    # Interpolate the required attributes
    texc, texd = dr.interpolate(uv_coords[None, ...], rast_out, uv_indices, rast_db=rast_out_db, diff_attrs='all')  # Interpolates the UV Coordinates
    normalsc, normalsd = dr.interpolate(vertex_normals[None, ...], rast_out, triface_indices, rast_db=rast_out_db, diff_attrs='all')   # Interpolates vertex normals
    vertexc, vertexd = dr.interpolate(vertex_positions_world[None, ...], rast_out, triface_indices, rast_db=rast_out_db, diff_attrs='all')   # Interpolates world space vertices

    # Apply Shading
    device = texc.device
    shader.set_device(device)

    # The colors are in rang [0,1], so we need to normalize them accordingly
    ambient_color, diffuse_color, specular_color = shader(points=vertexc, normals=normalsc, camera_position=torch.tensor([0., 0., -5.])[None, ...].to(device))

    if background is None:
        one_tensor = torch.ones((diffuse_color.shape[0], diffuse_color.shape[3], 1, 1), device=diffuse_color.device)
    else:
        one_tensor = background
    one_tensor_permuted = one_tensor.permute((0, 2, 3, 1)).contiguous()
    color = torch.where(mask, one_tensor_permuted, diffuse_color.float())  # [:, :, :, :-1]
    color = torch.cat((color, mask.float()), dim=-1)
    return color[:, :, :, :-1]


def intrinsic_to_projection(intrinsic_matrix):
    near, far = 0.1, 50.
    a, b = -(far + near) / (far - near), -2 * far * near / (far - near)
    projection_matrix = torch.tensor([
        intrinsic_matrix[0][0] / intrinsic_matrix[0][2],    0,                                                0, 0,
        0,                                                  -intrinsic_matrix[1][1] / intrinsic_matrix[1][2], 0, 0,
        0,                                                  0,                                                a, b,
        0,                                                  0,                                               -1, 0
    ]).float().reshape((4, 4))
    return projection_matrix


class DifferentiableRenderer(nn.Module):

    def __init__(self, resolution, mode='standard', shading=False, num_channels=3):
        super().__init__()
        self.glctx = dr.RasterizeGLContext()
        self.resolution = resolution
        self.num_channels = num_channels
        self.render_func_texture = render_with_texture
        self.shader = SoftPhongShader()
        self.shading = shading
        if mode == 'bounds':
            self.render_func_texture = render_in_bounds_with_texture

    def render_with_texture_map(self, vertex_positions, triface_indices, uv_coords, uv_indices, texture_image, ranges=None, background=None, resolution=None, vertex_normals=None, vertex_positions_world=None):
        if ranges is None:
            ranges = torch.tensor([[0, triface_indices.shape[0]]]).int()
        if resolution is None:
            resolution = self.resolution
        if self.shading:
            color = render_with_texture_shading(glctx=self.glctx, shader=self.shader, vertex_positions=vertex_positions, triface_indices=triface_indices,
                                                uv_coords=uv_coords, uv_indices=uv_indices, resolution=resolution, ranges=ranges, vertex_normals=vertex_normals,
                                                vertex_positions_world=vertex_positions_world, background=background)
        else:
            color = self.render_func_texture(self.glctx, vertex_positions, triface_indices, uv_coords, texture_image, uv_indices, resolution, ranges, background)
        return color[:, :, :, :self.num_channels]
