import random
import torch
import pickle
import numpy as np
from glob import glob
from util.misc import get_random_perspective_view, load_mesh, get_orthographic_view


class FaceMesh(torch.utils.data.Dataset):

    def __init__(self, w_codes_pth, mesh_path, flame_params, verts_pth, deca_warped_path, cam_mode='orthographic', viewpoint_x=None, mode='train'):
        super().__init__()
        self.device = None
        self.mode = mode
        self.cam_mode = cam_mode
        self.viewpoint_x = viewpoint_x
        _, faces, uvs, uv_indices = load_mesh(mesh_path)
        self.faces = torch.from_numpy(faces).int()
        self.uvs = torch.from_numpy(uvs).float()
        self.uvs = torch.cat([self.uvs[:, 0:1], 1 - self.uvs[:, 1:2]], dim=1)
        self.uv_indices = torch.from_numpy(uv_indices).int()

        # The list of latent codes
        self.w_codes = np.array(sorted(glob(w_codes_pth + '*.pt')))

        # Get the tform from deca
        with open(deca_warped_path, 'rb') as f:
            data_dict = pickle.load(f)
        self.tform = torch.tensor(data_dict['tform'].params).float()

        # Get the vertices
        with open(verts_pth, 'rb') as f:
            verts = pickle.load(f)
        self.vertices = torch.from_numpy(verts).float()

        # Get Flame Vertices
        with open(flame_params, 'rb') as f:
            flame_params = pickle.load(f)
        self.flame_params = flame_params

    def set_device(self, device):
        self.device = device
        self.vertices = self.vertices.to(device)
        self.uvs = self.uvs.to(device)
        self.faces = self.faces.to(device)
        self.uv_indices = self.uv_indices.to(device)

    def __len__(self):
        return len(self.w_codes)

    def __getitem__(self, index):
        projection_matrix, view_matrix = None, None
        if self.cam_mode == 'perspective':
            projection_matrix, view_matrix = get_random_perspective_view()
        elif self.cam_mode == 'orthographic':
            if self.mode == 'val':
                pos_x = 0
            else:
                pos_x = random.uniform(-2, 2)
            pos_y = 0
            if self.viewpoint_x is not None:
                projection_matrix, view_matrix = get_orthographic_view(pos_x=self.viewpoint_x[index], pos_y=pos_y)
            else:
                projection_matrix, view_matrix = get_orthographic_view(pos_x=pos_x, pos_y=pos_y)
        w_code = torch.load(self.w_codes[index])

        return {
            'name': f"{index:06d}",
            'w_code': w_code,
            'flame_shape': torch.from_numpy(self.flame_params['shape']).float(),
            'flame_exp': torch.from_numpy(self.flame_params['expression']).float(),
            'flame_pose': torch.from_numpy(self.flame_params['pose']).float(),
            'flame_tex': torch.from_numpy(self.flame_params['tex']).float(),
            'flame_cam': torch.from_numpy(self.flame_params['camera']).float(),
            'flame_light': torch.from_numpy(self.flame_params['light']).float(),
            "projection_matrix": projection_matrix,
            "view_matrix": view_matrix,
        }
