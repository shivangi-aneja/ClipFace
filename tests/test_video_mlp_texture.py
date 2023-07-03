import os
import torch
import yaml
import pickle
from glob import glob
import torch.nn as nn
import torchvision.utils
from omegaconf import OmegaConf
from model.mappers.mlp import Mapper
from model.flame.flame_model import FLAME
from model.stylegan_ada.generator import Generator
from util.misc import load_mesh, get_orthographic_view, transform_points
from model.renderer.differentiable_renderer import DifferentiableRenderer, transform_pos_mvp

image_size = 512
deca_size = 224
custom_scale_factor = 2.0
custom_scale_factor_image = 1024 / image_size
projection_matrix, view_matrix = get_orthographic_view(pos_x=0, pos_y=0)


def process_rendering(vertices, camera, tform):
    trans_verts = vertices[:, :2] + camera[1:]
    trans_verts = torch.cat([trans_verts, vertices[:, 2:]], dim=1)
    scaled_verts = custom_scale_factor * trans_verts * camera[0]
    vertices_mvp = transform_pos_mvp(scaled_verts, torch.matmul(projection_matrix, view_matrix).to(torch.device("cuda:0")).unsqueeze(0))
    points_scale = [deca_size, deca_size]  # Increases scale and shifts right + bottom for large values
    h, w = [custom_scale_factor_image * image_size, custom_scale_factor_image * image_size]  # Increases scale and shifts right + bottom for smaller values
    tform_1 = torch.inverse(tform[None, ...]).transpose(1, 2).to(torch.device("cuda:0"))
    vertices_mvp = transform_points(vertices_mvp.unsqueeze(0), tform_1, points_scale, [h, w])[0]
    return vertices_mvp


def visualize_rendering():
    with open('config/clipface.yaml', "r") as yamlfile:
        conf = yaml.load(yamlfile, Loader=yaml.FullLoader)
        config = OmegaConf.create(conf)

    # The expression codes
    flame_codes_pth = sorted(glob(config.exp_codes_pth + '*.pkl'))
    exp_codes = []
    pose_codes = []
    for exp_code_pth in flame_codes_pth:
        with open(exp_code_pth, 'rb') as f:
            flame_params_frames = pickle.load(f)
            exp_codes.append(torch.from_numpy(flame_params_frames['exp']).float())
            pose_codes.append(torch.from_numpy(flame_params_frames['pose']).float())

    img_latent_pth = 'clip_latents_val/w_001368_5907.pt'  # TODO: Update checkpoint path here
    mapper_ckpt = 'checkpoints/texture_video/angry.pt'    # TODO: Update texture latent code path here

    w_init = torch.load(img_latent_pth, map_location=torch.device("cpu")).unsqueeze(0)
    _flame = FLAME(config)
    _renderer = DifferentiableRenderer(image_size, mode='standard', num_channels=3)
    _G = Generator(z_dim=config.latent_dim, w_dim=config.latent_dim,
                   w_num_layers=config.num_mapping_layers, img_resolution=config.image_size,
                   img_channels=3, synthesis_layer=config.generator)

    # Load the pretrained checkpoints
    _G.load_state_dict(torch.load(config.pretrained_stylegan_pth, map_location=torch.device("cpu")))
    _G.eval()

    # Mapper for expression conditioned textures
    _texture_mapper_list = []
    mapper_state_dict = torch.load(mapper_ckpt)
    for level in range(18):
        mapper = Mapper(z_dim=config.latent_dim + config.expression_params + 6, w_dim=config.latent_dim, num_layers=1)
        mapper.load_state_dict(mapper_state_dict[f"level_{level}"])
        mapper = mapper.to(torch.device("cuda:0"))
        _texture_mapper_list.append(mapper)

    flame_params = "data/clip/flame_params.pkl"
    deca_warped_path = 'data/clip/sample_deca.pkl'

    # The fitted flame parameters for neutral mesh
    with open(flame_params, 'rb') as f:
        flame_params = pickle.load(f)

    # Deca Transformation parameters
    with open(deca_warped_path, 'rb') as f:
        data_dict = pickle.load(f)
    tform = torch.tensor(data_dict['tform'].params).float()

    # Load template Mesh
    _, faces, uvs, uv_indices = load_mesh("data/flame/head_template.obj")
    faces = torch.from_numpy(faces).int()
    uvs = torch.from_numpy(uvs).float()
    uvs = torch.cat([uvs[:, 0:1], 1 - uvs[:, 1:2]], dim=1)
    uv_indices = torch.from_numpy(uv_indices).int()

    # Set device
    _flame = _flame.to(torch.device("cuda:0"))
    _renderer = _renderer.to(torch.device("cuda:0"))
    _G = _G.to(torch.device("cuda:0"))
    faces = faces.to(torch.device("cuda:0"))
    uvs = uvs.to(torch.device("cuda:0"))
    uv_indices = uv_indices.to(torch.device("cuda:0"))
    camera = torch.from_numpy(flame_params['camera']).float().to(torch.device("cuda:0"))

    # The softmax expression deltas
    l2_loss = nn.MSELoss(reduce=False, reduction='none')
    exp_neutral = torch.zeros(1, config.expression_params).float()
    softmax_scores_unnormalized = []
    for exp_code, pose_code in zip(exp_codes, pose_codes):
        exp_pose_neutral = torch.cat((exp_neutral, pose_code[:, :3], torch.zeros(1, 3)), dim=1)
        exp_pose_modified = torch.cat((exp_code, pose_code), dim=1)
        delta = l2_loss(exp_pose_modified, exp_pose_neutral).sum()
        softmax_scores_unnormalized.append(delta)
    softmax_scores_unnormalized = torch.as_tensor(softmax_scores_unnormalized)
    normalized = (softmax_scores_unnormalized - min(softmax_scores_unnormalized)) / (max(softmax_scores_unnormalized) - min(softmax_scores_unnormalized))
    normalized_scores = normalized + 1e-6

    # Pedict and save the temporal frames
    with torch.no_grad():
        frame_pth = f"vis/{img_latent_pth.split('/')[-1].split('.')[0]}_frames"
        os.makedirs(frame_pth, exist_ok=True)
        for idx, (exp_code, pose_code) in enumerate(zip(exp_codes, pose_codes)):
            shape_params = torch.from_numpy(flame_params['shape']).unsqueeze(0).float()
            w_offsets = None
            for idx2, mapper in enumerate(_texture_mapper_list):
                mapper_input = torch.cat((w_init[:, idx2, :], exp_code, pose_code), dim=-1).to(torch.device("cuda:0"))
                w_offset_layer = mapper(mapper_input).unsqueeze(1)
                if w_offsets is None:
                    w_offsets = w_offset_layer
                else:
                    w_offsets = torch.cat((w_offsets, w_offset_layer), dim=1)
            w_offsets = w_offsets
            w_vid = w_init.to(torch.device("cuda:0")) + normalized_scores[idx] * w_offsets

            predicted_texture = _G.synthesis(w_vid, noise_mode='const')

            # Get the new vertices
            vertices_new_geo = _flame(shape_params=shape_params.to(torch.device("cuda:0")), expression_params=exp_code.to(torch.device("cuda:0")), pose_params=pose_code.to(torch.device("cuda:0")), eye_pose_params=None)[0].contiguous()
            vertices_mvp_new_geo = process_rendering(vertices_new_geo, camera, tform)

            # Render geometry with Shading
            img_new_geo = _renderer.render_with_texture_map(vertex_positions=vertices_mvp_new_geo, triface_indices=faces, uv_coords=uvs, uv_indices=uv_indices, texture_image=predicted_texture).permute((0, 3, 1, 2))
            torchvision.utils.save_image(img_new_geo[0], f"{frame_pth}/{idx:00004d}.jpg", value_range=(-1, 1), normalize=True)


if __name__ == "__main__":
    visualize_rendering()
