import torch
import math
import os
import random
import torch_scatter
import numpy as np
import json
from pathlib import Path
from ballpark import business
from collections import OrderedDict
from util.camera import spherical_coord_to_cam, OrthographicCamera


def print_model_parameter_count(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in {type(model).__name__}: {business(count, precision=3, prefix=True)}")


def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen if t.requires_grad]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(e.outputs[0].shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def get_random_perspective_view():
    azimuth = math.pi + (random.random() - 0.5) * math.pi / 2
    elevation = math.pi / 2 + (random.random() - 0.5) * math.pi / 4
    perspective_cam = spherical_coord_to_cam(50, azimuth, elevation, cam_dist=1.25)
    projection_matrix = torch.from_numpy(perspective_cam.projection_mat()).float()
    view_matrix = torch.from_numpy(perspective_cam.view_mat()).float()
    return projection_matrix, view_matrix


def get_fixed_view():
    azimuth = math.pi
    # elevation = math.pi / 2 + 0.1
    elevation = math.pi / 2
    perspective_cam = spherical_coord_to_cam(50, azimuth, elevation, cam_dist=1.25)
    projection_matrix = torch.from_numpy(perspective_cam.projection_mat()).float()
    view_matrix = torch.from_numpy(perspective_cam.view_mat()).float()
    return projection_matrix, view_matrix


def get_fixed_view_test(view):
    azimuth = math.pi + (view - 0.5) * math.pi / 2
    elevation = math.pi / 2
    perspective_cam = spherical_coord_to_cam(50, azimuth, elevation, cam_dist=1.25)
    projection_matrix = torch.from_numpy(perspective_cam.projection_mat()).float()
    view_matrix = torch.from_numpy(perspective_cam.view_mat()).float()
    return projection_matrix, view_matrix


def get_orthographic_view(pos_x=0, pos_y=0, max_len=1):
    shape = (max_len * 2, max_len * 2)
    cam_dist = -5.0
    lookat = (0, 0, 0)
    orthograhic_cam = OrthographicCamera(size=shape, near=2.0, far=5000.0, position=(pos_x, pos_y, -cam_dist), clear_color=(1, 1, 1, 1), lookat=lookat, up=(0, 1, 0))
    projection_matrix = torch.from_numpy(orthograhic_cam.projection_mat()).float()
    view_matrix = torch.from_numpy(orthograhic_cam.view_mat()).float()
    return projection_matrix, view_matrix


def vertex_to_normals(vertices, faces):
    triangles = vertices[faces, :3]
    vector_0 = triangles[:, 1, :] - triangles[:, 0, :]
    vector_1 = triangles[:, 2, :] - triangles[:, 1, :]
    cross = torch.cross(vector_0, vector_1, dim=1)
    face_normals = torch.nn.functional.normalize(cross, p=2.0, dim=1)
    vertex_normals = torch.zeros((vertices.shape[0], 3), device=vertices.device)
    torch_scatter.scatter_mean(face_normals.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3), faces.reshape(-1), dim=0, out=vertex_normals)
    vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2.0, dim=1)
    vertex_normals = torch.cat([vertex_normals, torch.ones([vertex_normals.shape[0], 1], device=vertex_normals.device)], dim=1)
    return vertex_normals


def load_mesh(mesh_path):
    cvt = lambda x, t: [t(y) for y in x]
    mesh_text = Path(mesh_path).read_text().splitlines()
    vertices, indices, uvs, uv_indices = [], [], [], []
    for line in mesh_text:
        if line.startswith("v "):
            vertices.append(cvt(line.split(" ")[1:4], float))
        if line.startswith("vt "):
            uvs.append(cvt(line.split(" ")[1:], float))
        if line.startswith("f "):
            if '/' in line:
                indices.append([int(x.split('/')[0]) - 1 for x in line.split(' ')[1:]])
                uv_indices.append([int(x.split('/')[1]) - 1 for x in line.split(' ')[1:]])
            else:
                indices.append([int(x) - 1 for x in line.split(' ')[1:]])
    return np.array(vertices), np.array(indices), np.array(uvs), np.array(uv_indices)


def get_parameters_from_state_dict(state_dict, filter_key):
    new_state_dict = OrderedDict()
    for k in state_dict:
        if k.startswith(filter_key):
            new_state_dict[k.replace(filter_key + '.', '')] = state_dict[k]
    return new_state_dict


def copy_state_dict(cur_state_dict, pre_state_dict, prefix='', load_name=None):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None
    for k in cur_state_dict.keys():
        if load_name is not None:
            if load_name not in k:
                continue
        v = _get_params(k)
        try:
            if v is None:
                print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            print('copy param {} failed'.format(k))
            continue


def transform_points(points, tform, points_scale=None, out_scale=None):
    points_2d = points[:, :, :2]
    # 'input points must use original range'
    if points_scale:
        assert points_scale[0] == points_scale[1]
        points_2d = (points_2d * 0.5 + 0.5) * points_scale[0]

    batch_size, n_points, _ = points.shape
    trans_points_2d = torch.bmm(torch.cat([points_2d, torch.ones([batch_size, n_points, 1], device=points.device, dtype=points.dtype)], dim=-1), tform)
    if out_scale:  # h,w of output image size
        trans_points_2d[:, :, 0] = trans_points_2d[:, :, 0] / out_scale[1] * 2 - 1
        trans_points_2d[:, :, 1] = trans_points_2d[:, :, 1] / out_scale[0] * 2 - 1
    trans_points = torch.cat([trans_points_2d[:, :, :2], points[:, :, 2:]], dim=-1)
    return trans_points


def write_json_data(dic, file_name, mode='a'):
    with open(file_name, mode) as output_file:
        json.dump(dic, output_file)
        output_file.write(os.linesep)


def read_json_data(file_name):
    with open(file_name) as f:
        article_list = [json.loads(line) for line in f]
        return article_list


def tensor2im(var, normalize_negative=True):
    var = var.cpu().detach().numpy()
    if normalize_negative:
        var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return var
