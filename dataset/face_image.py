import cv2
import torch
import numpy as np
from glob import glob
from util.misc import load_mesh
from torch.utils.data import DataLoader
from util.misc import get_orthographic_view


class FaceImage(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        self.image_size = config.image_size
        self.deca_size = config.deca_size
        self.scale = 1.25

        mesh_path = config.flame_obj
        vertices, faces, uvs, uv_indices = load_mesh(mesh_path)
        vertex_bounds = (vertices.min(axis=0), vertices.max(axis=0))
        self.vertices = vertices - (vertex_bounds[0] + vertex_bounds[1]) / 2
        self.vertices = self.vertices / (vertex_bounds[1] - vertex_bounds[0]).max()
        self.vertices = torch.from_numpy(self.vertices).float()
        self.faces = torch.from_numpy(faces).int()
        self.uvs = torch.from_numpy(uvs).float()
        self.uvs = torch.cat([self.uvs[:, 0:1], 1 - self.uvs[:, 1:2]], dim=1)
        self.uv_indices = torch.from_numpy(uv_indices).int()

        self.image_paths = np.array(sorted(glob(config.ffhq_dir_512 + '/*.png')))
        self.vertices_paths = np.array(sorted(glob(config.ffhq_verts_dir + '/*.pt')))

    def set_renderer(self, renderer):
        self.renderer = renderer

    def __len__(self):
        return len(self.image_paths)

    def set_device(self, device):
        self.device = device
        self.vertices = self.vertices.to(device)
        self.uvs = self.uvs.to(device)
        self.faces = self.faces.to(device)
        self.uv_indices = self.uv_indices.to(device)

    def __getitem__(self, index):

        projection_matrix, view_matrix = get_orthographic_view()

        # # Image Paths
        image_path = self.image_paths[index]
        vertex_path = self.vertices_paths[index]

        # Corresponding Real Image for StyleGAN
        image_cv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_cv, (self.image_size, self.image_size), cv2.INTER_AREA)
        image = torch.from_numpy(image_np).float() / 127.5 - 1

        # Load the vertex
        vertex = torch.load(vertex_path, map_location='cpu')

        return {
            "name": f"{index:06d}",
            "image": image.permute(2, 0, 1),
            "projection_matrix": projection_matrix,
            "view_matrix": view_matrix,
            "vertices":  vertex,
        }
