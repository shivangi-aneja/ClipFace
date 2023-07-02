import hydra
from abc import ABC
from pathlib import Path
import torch.multiprocessing
import pytorch_lightning as pl
from trainer import create_trainer
from util.stylegan_utils import *
from model.mappers.mlp import Mapper
from dataset.face_mesh import FaceMesh
from criteria.clip_loss import ClipLoss
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model.flame.flame_model import FLAME
from model.stylegan_ada.generator import Generator
from pytorch_lightning.utilities import rank_zero_only
from util.misc import transform_points, get_parameters_from_state_dict
from model.renderer.differentiable_renderer import DifferentiableRenderer, transform_pos_mvp

torch.multiprocessing.set_sharing_strategy('file_system')  # a fix for the "OSError: too many files" exception
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True

class StyleGANOptimizer(pl.LightningModule, ABC):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.R = None
        self.clip_loss = ClipLoss(config=config)
        self.flame = FLAME(config)
        self.G = Generator(z_dim=config.latent_dim, w_dim=config.latent_dim,
                           w_num_layers=config.num_mapping_layers, img_resolution=config.image_size,
                           img_channels=3, synthesis_layer=config.generator)
        self.train_set = FaceMesh(w_codes_pth=config.w_train_pth, mesh_path="data/flame/head_template.obj",
                                  flame_params="data/clip/flame_params.pkl", verts_pth='data/clip/verts.pkl',
                                  deca_warped_path='data/clip/sample_deca.pkl')
        self.val_set = FaceMesh(w_codes_pth=config.w_val_pth, mesh_path="data/flame/head_template.obj",
                                flame_params="data/clip/flame_params.pkl", verts_pth='data/clip/verts.pkl',
                                deca_warped_path='data/clip/sample_deca.pkl', mode='val')
        self.automatic_optimization = False
        if config.batch_gpu is None:
            config.batch_gpu = config.batch_size
        print(f"batch_size = {config.batch_size} / {config.batch_gpu}")
        assert config.batch_size >= config.batch_gpu and config.batch_size % config.batch_gpu == 0
        self.G.load_state_dict(torch.load(config.pretrained_stylegan_pth, map_location=torch.device("cpu")))
        self.G.eval()

        state_dict = torch.load(config.pretrain_mapper, map_location=torch.device("cpu"))["state_dict"]

        # Mapper for expression conditioned textures
        self.texture_mapper_list = []
        for i in range(18):
            mapper = Mapper(z_dim=config.latent_dim, w_dim=config.latent_dim, num_layers=4)
            mapper.load_state_dict(get_parameters_from_state_dict(state_dict, "texture_mapper"))
            self.texture_mapper_list.append(mapper)

    def configure_optimizers(self):
        trainable_params = []
        for mapper in self.texture_mapper_list:
            trainable_params += list(mapper.parameters())
        code_opt = torch.optim.Adam([{'params': trainable_params, 'lr': self.config.lr_tex}],
                                    lr=self.config.lr_tex)
        return code_opt

    def forward(self, batch):

        self.G.eval()
        # Get the new texture
        w_offsets = None
        for idx, mapper in enumerate(self.texture_mapper_list):
            w_offset_layer = mapper(batch['w_code'][:, idx, :])
            if w_offsets is None:
                w_offsets = w_offset_layer
            else:
                w_offsets = torch.cat((w_offsets, w_offset_layer), dim=0)
        w_offsets = w_offsets.unsqueeze(0)

        w = batch['w_code'] + w_offsets
        init_texture = self.G.synthesis(batch['w_code'], noise_mode='const')
        predicted_texture = self.G.synthesis(w, noise_mode='const')

        vertices = self.flame(shape_params=batch['flame_shape'], expression_params=batch['flame_exp'],
                              pose_params=batch['flame_pose'])
        init_img = self.render(init_texture, vertices, batch)
        pred_img = self.render(predicted_texture, vertices, batch)

        return {'tex': predicted_texture, 'init_img': init_img,  'pred_img': pred_img}

    def training_step(self, batch, batch_idx):
        code_opt = self.optimizers()
        results = self.forward(batch)
        clip_loss = self.clip_loss(results['pred_img'], results['init_img'])
        self.log(f"train/clip_loss", clip_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.manual_backward(clip_loss)
        step(code_opt, self.texture_mapper_list)
        torch.cuda.empty_cache()

    def render(self, texture_image, flame_verts, batch):
        vertices, uvs = [], []
        faces, uv_indices = [], []
        num_vertex, num_uvs = 0, 0
        start_index = 0
        ranges = []
        for i in range(self.config.batch_size):
            # Translate and scale according to Flame camera parameters
            custom_scale_factor = 2.0
            custom_scale_factor_image = 1024 / self.config.image_size
            camera = batch['flame_cam'][i]
            trans_verts = flame_verts[i][:, :2] + camera[1:]
            trans_verts = torch.cat([trans_verts, flame_verts[i][:, 2:]], dim=1)
            scaled_verts = custom_scale_factor * trans_verts * camera[0]

            # Apply model-view and projection transform.
            projection_matrix, view_matrix = batch['projection_matrix'][i], batch['view_matrix'][i]
            vertices_mvp = transform_pos_mvp(scaled_verts, torch.matmul(projection_matrix, view_matrix).to(self.device).unsqueeze(0))

            # Apply scaling to fit-in the image size
            points_scale = [self.config.deca_size, self.config.deca_size]
            h, w = [custom_scale_factor_image * self.config.image_size, custom_scale_factor_image * self.config.image_size]  # Increases scale and shifts right + bottom for smaller values
            tform = torch.inverse(self.train_set.tform[None, ...]).transpose(1, 2).to(self.device)
            vertices_mvp = transform_points(vertices_mvp.unsqueeze(0), tform, points_scale, [h, w])[0]

            vertices.append(vertices_mvp)
            uvs.append(self.train_set.uvs)
            faces.append(self.train_set.faces + num_vertex)
            uv_indices.append(self.train_set.uv_indices + num_uvs)
            num_vertex += self.train_set.vertices.shape[0]
            num_uvs += self.train_set.uvs.shape[0]
            ranges.append(torch.tensor([start_index, self.train_set.faces.shape[0]]).int())
            start_index += self.train_set.faces.shape[0]
        vertices, uvs, faces, uv_indices, ranges = torch.cat(vertices, 0), torch.cat(uvs, 0), torch.cat(faces, 0), torch.cat(uv_indices, 0), torch.stack(ranges, dim=0)
        return self.R.render_with_texture_map(vertices, faces, uvs, uv_indices, texture_image, ranges=ranges).permute((0, 3, 1, 2)).contiguous()

    def validation_step(self, batch, batch_idx):
        results = self.forward(batch)
        clip_loss = self.clip_loss(results['pred_img'], results['init_img'])
        self.log(f"val/clip_loss", clip_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        torch.cuda.empty_cache()

    @rank_zero_only
    def validation_epoch_end(self, _val_step_outputs):
        odir_samples, odir_grid, odir_texture, odir_ckpts = self.create_directories()
        self.export_images(odir_grid, odir_samples, odir_texture, odir_ckpts)

    def create_directories(self):
        output_dir_grid = Path(f'{self.config.base_dir}/runs/{self.config.experiment}/grid/')
        output_dir_texture = Path(f'{self.config.base_dir}/runs/{self.config.experiment}/texture/{self.global_step:06d}')
        output_dir_samples = Path(f'{self.config.base_dir}/runs/{self.config.experiment}/images/{self.global_step:06d}')
        output_dir_ckpts = Path(f'{self.config.base_dir}/runs/{self.config.experiment}/ckpts/')
        for odir in [output_dir_grid, output_dir_samples, output_dir_texture, output_dir_ckpts]:
            odir.mkdir(exist_ok=True, parents=True)
        return output_dir_samples, output_dir_grid, output_dir_texture, output_dir_ckpts

    def export_images(self, odir_grid, odir_samples, odir_texture, odir_ckpts):
        vis_generated_images = []
        with torch.no_grad():
            for iter_idx, batch in enumerate(self.val_dataloader()):
                if iter_idx < self.config.num_vis_images // self.config.batch_size:
                    batch['flame_shape'] = batch['flame_shape'].to(self.device)
                    batch['flame_exp'] = batch['flame_exp'].to(self.device)
                    batch['flame_pose'] = batch['flame_pose'].to(self.device)
                    batch['flame_cam'] = batch['flame_cam'].to(self.device)
                    results = self.forward(batch=batch)
                    rendering_clip = results['pred_img']
                    for b_idx in range(self.config.batch_size):
                        save_image(rendering_clip[b_idx], odir_samples / f"pred_{batch['name'][b_idx]}_{iter_idx}_{b_idx}.jpg", value_range=(-1, 1), normalize=True)
                        save_image(results['tex'][b_idx], odir_texture / f"tex_{batch['name'][b_idx]}_{iter_idx}_{b_idx}.jpg", value_range=(-1, 1), normalize=True)
                    vis_generated_images.append(rendering_clip)
        torch.cuda.empty_cache()
        vis_generated_images = torch.cat(vis_generated_images, dim=0)
        save_image(vis_generated_images, odir_grid / f"{self.global_step:06d}.png", nrow=4, value_range=(-1, 1), normalize=True)
        mapper_state_dict = {}
        for level, mapper in enumerate(self.texture_mapper_list):
            mapper_state_dict[f"level_{level}"] = mapper.state_dict()
        torch.save(mapper_state_dict, odir_ckpts / f"{self.global_step:06d}.pt")

    def train_dataloader(self):
        return DataLoader(self.train_set, self.config.batch_size, shuffle=False, pin_memory=False, drop_last=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.config.batch_size, shuffle=False, drop_last=True, num_workers=self.config.num_workers)

    def on_train_start(self):
        self.run_post_device_setup()

    def on_validation_start(self):
        self.run_post_device_setup()

    def run_post_device_setup(self):
        if self.R is None:
            self.R = DifferentiableRenderer(self.config.image_size, "standard")
        self.R.to(self.device)
        self.flame.to(self.device)
        self.G.to(self.device)
        for mapper in self.texture_mapper_list:
            mapper.to(self.device)
        self.train_set.set_device(self.device)
        self.val_set.set_device(self.device)
        self.clip_loss.set_device(self.device)


def step(opt, module_list):
    for module in module_list:
        for param in module.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        opt.step()


@hydra.main(config_path='../../config', config_name='clipface')
def main(config):
    trainer = create_trainer("StyleGANClipMLP", config)
    model = StyleGANOptimizer(config)
    trainer.fit(model)


if __name__ == '__main__':
    main()
