"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
"""
# Modified from smplx code for FLAME
import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
from .lbs import lbs


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class FLAME(nn.Module):
    """
        Given flame parameters this class generates a differentiable FLAME function
        which outputs the mesh vertices
    """

    def __init__(self, config):
        super(FLAME, self).__init__()
        print("Creating the FLAME Decoder")
        with open(config.flame_model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))

        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))

        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:, :, :config.shape_params], shapedirs[:, :, 300:300+config.expression_params]], 2)
        self.register_buffer('shapedirs', shapedirs)

        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))

        # The joint regressor
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long(); parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose, requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose, requires_grad=False))

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None):
        """
            Input:
                shape_params: N X number of shape parameters (100)
                expression_params: N X number of expression parameters (50)
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose
        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat([pose_params[:, :3], self.neck_pose.expand(batch_size, -1), pose_params[:, 3:], eye_pose_params.expand(batch_size, -1)], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        # import ipdb; ipdb.set_trace()
        vertices, _ = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)

        return vertices


class FLAMETex(nn.Module):
    """
        Current FLAME texture are adapted from BFM Texture Model
    """

    def __init__(self, config):
        super(FLAMETex, self).__init__()
        tex_params = config.tex_params
        tex_space = np.load(config.tex_space_path)
        texture_mean = tex_space['mean'].reshape(1, -1)
        texture_basis = tex_space['tex_dir'].reshape(-1, 200)
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :tex_params]).float()[None, ...]
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

    def set_device(self, device):
        self.texture_mean = self.texture_mean.to(device)
        self.texture_basis = self.texture_basis.to(device)

    def forward(self, texcode):
        texture = self.texture_mean + (self.texture_basis*texcode[:, None, :]).sum(-1)
        texture = texture.reshape(512, 512, 3)[..., [2, 1, 0]]  # BGR to RGB
        texture = texture / 127.5 - 1
        return texture


