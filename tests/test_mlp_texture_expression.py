import torch
import yaml
import pickle
from omegaconf import OmegaConf
from model.mappers.mlp import Mapper
from util.misc import vertex_to_normals
from torchvision.utils import save_image
from model.flame.flame_model import FLAME
from model.stylegan_ada.generator import Generator
from util.misc import get_orthographic_view, transform_points, load_mesh
from model.renderer.differentiable_renderer import DifferentiableRenderer, transform_pos_mvp

image_size = 512
deca_size = 224
batch_size = 4
custom_scale_factor = 2.0
custom_scale_factor_image = 1024 / image_size
projection_matrix, view_matrix = get_orthographic_view(pos_x=0, pos_y=0)


def process_rendering(vertices, camera, tform, projection_matrix, view_matrix):
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

    # Initialize the models
    flame_params = "data/clip/flame_params.pkl"
    deca_warped_path = 'data/clip/sample_deca.pkl'
    mapper_ckpt = 'checkpoints/texture_expression/angry.pt'    # TODO: Update checkpoint path here
    w_init_pth = 'clip_latents_val/w_000049_7548.pt'        # TODO: Update texture latent code path here
    _flame = FLAME(config)
    _renderer = DifferentiableRenderer(image_size, mode='standard', num_channels=3, shading=False)
    _renderer_geo = DifferentiableRenderer(image_size, mode='standard', num_channels=3, shading=True)
    _G = Generator(z_dim=config.latent_dim, w_dim=config.latent_dim, w_num_layers=config.num_mapping_layers,
                   img_resolution=config.image_size, img_channels=3, synthesis_layer=config.generator)
    _texture_mapper_list = []
    _geometry_mapper = Mapper(z_dim=config.latent_dim, w_dim=config.expression_params, num_layers=4)

    # Load the pretrained checkpoints
    mapper_state_dict = torch.load(mapper_ckpt)
    for level in range(18):
        mapper = Mapper(z_dim=config.latent_dim, w_dim=config.latent_dim, num_layers=4)
        mapper.load_state_dict(mapper_state_dict[f"level_{level}"])
        mapper = mapper.to(torch.device("cuda:0"))
        _texture_mapper_list.append(mapper)
    _geometry_mapper.load_state_dict(mapper_state_dict["geo"])
    _G.load_state_dict(torch.load(config.pretrained_stylegan_pth, map_location=torch.device("cpu")))
    _G.eval()
    w_init_code = torch.load(w_init_pth)

    with open(flame_params, 'rb') as f:
        flame_params = pickle.load(f)

    with open(deca_warped_path, 'rb') as f:
        data_dict = pickle.load(f)
    tform = torch.tensor(data_dict['tform'].params).float()

    # Load Mesh
    _, faces, uvs, uv_indices = load_mesh("data/flame/head_template.obj")
    faces = torch.from_numpy(faces).int()
    uvs = torch.from_numpy(uvs).float()
    uvs = torch.cat([uvs[:, 0:1], 1 - uvs[:, 1:2]], dim=1)
    uv_indices = torch.from_numpy(uv_indices).int()

    # Set device
    _flame = _flame.to(torch.device("cuda:0"))
    _G = _G.to(torch.device("cuda:0"))
    _renderer = _renderer.to(torch.device("cuda:0"))
    _renderer_geo = _renderer_geo.to(torch.device("cuda:0"))
    _geometry_mapper = _geometry_mapper.to(torch.device("cuda:0"))
    w_init_code = w_init_code.to(torch.device("cuda:0")).unsqueeze(0)
    faces = faces.to(torch.device("cuda:0"))
    uvs = uvs.to(torch.device("cuda:0"))
    uv_indices = uv_indices.to(torch.device("cuda:0"))
    camera = torch.from_numpy(flame_params['camera']).float().to(torch.device("cuda:0"))
    shape_params = torch.from_numpy(flame_params['shape']).float().to(torch.device("cuda:0")).unsqueeze(0)
    expression_params = torch.from_numpy(flame_params['expression']).float().to(torch.device("cuda:0")).unsqueeze(0)
    pose_params = torch.from_numpy(flame_params['pose']).float().to(torch.device("cuda:0")).unsqueeze(0)

    with torch.no_grad():
        w_offsets = None
        for idx, mapper in enumerate(_texture_mapper_list):
            w_offset_layer = mapper(w_init_code[:, idx, :])
            if w_offsets is None:
                w_offsets = w_offset_layer
            else:
                w_offsets = torch.cat((w_offsets, w_offset_layer), dim=0)
        w_offsets = w_offsets.unsqueeze(0)

        w = w_init_code + w_offsets

        geo_offset = _geometry_mapper(w.mean(dim=1))
        expression_params += geo_offset
        init_texture = _G.synthesis(w_init_code, noise_mode='const')
        predicted_texture = _G.synthesis(w, noise_mode='const')

        pred_vertices = _flame(shape_params=shape_params, expression_params=expression_params, pose_params=pose_params).squeeze()
        vertex_normals = vertex_to_normals(pred_vertices, faces.long())[..., :3].contiguous()
        vertices_mvp = process_rendering(pred_vertices, camera, tform, projection_matrix, view_matrix).contiguous()
        pred_init = _renderer.render_with_texture_map(vertices_mvp, faces, uvs, uv_indices, init_texture, background=None).permute(0, 3, 1, 2)
        pred_final = _renderer.render_with_texture_map(vertices_mvp, faces, uvs, uv_indices, predicted_texture, background=None).permute(0, 3, 1, 2)
        pred_geo = _renderer_geo.render_with_texture_map(vertex_positions=vertices_mvp, triface_indices=faces,
                                                         uv_coords=uvs,
                                                         uv_indices=uv_indices,
                                                         vertex_positions_world=pred_vertices.contiguous(),
                                                         vertex_normals=vertex_normals, texture_image=None, ranges=None,
                                                         resolution=image_size).permute((0, 3, 1, 2))
        prediction = torch.cat((pred_init, pred_final, pred_geo), dim=3)
        save_image(prediction, f"prediction.jpg", value_range=(-1, 1), normalize=True)


if __name__ == "__main__":
    visualize_rendering()
