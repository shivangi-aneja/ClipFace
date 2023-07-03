import os
import hydra
import pickle
from abc import ABC
import torch.nn as nn
from glob import glob
from pathlib import Path
import torch.multiprocessing
import pytorch_lightning as pl
from trainer import create_trainer
from util.stylegan_utils import *
from model.mappers.mlp import Mapper
from criteria.clip_loss import ClipLoss
from util.misc import transform_points
from dataset.face_mesh_video import FaceMesh
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model.flame.flame_model import FLAME
from model.stylegan_ada.generator import Generator
from pytorch_lightning.utilities import rank_zero_only
from model.renderer.differentiable_renderer import DifferentiableRenderer, transform_pos_mvp

torch.multiprocessing.set_sharing_strategy('file_system')  # a fix for the "OSError: too many files" exception
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class StyleGANOptimizer(pl.LightningModule, ABC):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.R = None
        self.flame = FLAME(config)
        self.clip_loss = ClipLoss(config=config)
        self.l2_loss = nn.MSELoss(reduce=False, reduction='none')
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

        # The list of expression codes
        exp_codes_pth = sorted(glob(config.exp_codes_pth + '*.pkl'))
        exp_codes = []
        pose_codes = []
        for exp_code_pth in exp_codes_pth:
            with open(exp_code_pth, 'rb') as f:
                flame_params_frames = pickle.load(f)
                exp_codes.append(torch.from_numpy(flame_params_frames['exp']).float())
                pose_codes.append(torch.from_numpy(flame_params_frames['pose']).float())
        self.exp_codes = torch.stack(exp_codes)
        self.pose_codes = torch.stack(pose_codes)
        self.num_frames = len(exp_codes)

        # The softmax expression deltas
        exp_neutral = torch.zeros(1, config.expression_params).float()
        softmax_scores_unnormalized = []
        for exp_code, pose_code in zip(exp_codes, pose_codes):
            exp_pose_neutral = torch.cat((exp_neutral, pose_code[:, :3], torch.zeros(1, 3)), dim=1)
            exp_pose_modified = torch.cat((exp_code, pose_code), dim=1)
            delta = self.l2_loss(exp_pose_modified, exp_pose_neutral).sum()
            softmax_scores_unnormalized.append(delta)
        softmax_scores_unnormalized = torch.as_tensor(softmax_scores_unnormalized)
        normalized = (softmax_scores_unnormalized - min(softmax_scores_unnormalized)) / (max(softmax_scores_unnormalized) - min(softmax_scores_unnormalized))
        self.normalized_scores = normalized.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) + 1e-6  # For numerical stability

        # Generator
        self.G.load_state_dict(torch.load(config.pretrained_stylegan_pth, map_location=torch.device("cpu")))
        self.G.eval()

        # Mapper for expression conditioned textures
        self.texture_mapper_list = []
        state_dict = torch.load(config.pretrain_video_mapper_pth, map_location=torch.device("cpu"))
        for i in range(18):
            mapper = Mapper(z_dim=config.latent_dim + config.expression_params + 6, w_dim=config.latent_dim, num_layers=1)
            mapper.load_state_dict(state_dict["texture_mapper"])
            self.texture_mapper_list.append(mapper)

    def configure_optimizers(self):
        trainable_texure_params = []
        for mapper in self.texture_mapper_list:
            trainable_texure_params += list(mapper.parameters())
        code_opt = torch.optim.Adam([{'params': trainable_texure_params, 'lr': self.config.lr_tex}], lr=self.config.lr_tex)
        return code_opt

    def gather_temporal_frames(self, w_code, batch):
        tex_frames, img_frames, vtx_frames = [], [], []
        predicted_tex_frames = self.G.synthesis(w_code.squeeze(), noise_mode='const')
        for idx in range(self.num_frames):
            expression_params = self.exp_codes[idx].repeat(self.config.batch_size, 1)
            pose_params = self.pose_codes[idx].repeat(self.config.batch_size, 1)
            vertices = self.flame(shape_params=batch['flame_shape'], expression_params=expression_params, pose_params=pose_params)
            img = self.render(predicted_tex_frames[idx].repeat(self.config.batch_size, 1, 1, 1), vertices, batch)
            tex_frames.append(predicted_tex_frames[idx])
            img_frames.append(img)
            vtx_frames.append(vertices)
        return {'tex_frames': tex_frames, 'img_frames': img_frames, 'vtx_frames': vtx_frames}

    def forward(self, batch):

        with torch.no_grad():
            w_init = batch['w_code'].unsqueeze(0).repeat(1, self.num_frames, 1, 1)
            results_init = self.gather_temporal_frames(w_init, batch)

        # Get the new texture
        w_offsets = None
        for idx, mapper in enumerate(self.texture_mapper_list):
            mapper_input = torch.cat((batch['w_code'][:, idx, :].unsqueeze(0).repeat(self.num_frames, 1, 1), self.exp_codes, self.pose_codes), dim=-1).squeeze()
            w_offset_layer = mapper(mapper_input).unsqueeze(1)
            if w_offsets is None:
                w_offsets = w_offset_layer
            else:
                w_offsets = torch.cat((w_offsets, w_offset_layer), dim=1)
        w_offsets = w_offsets.unsqueeze(0)

        w_vid = w_init + self.normalized_scores * w_offsets
        results_final = self.gather_temporal_frames(w_vid, batch)
        return results_init, results_final

    def training_step(self, batch, batch_idx):
        code_opt = self.optimizers()
        results_init, results_final = self.forward(batch)
        clip_loss = 0
        for idx in range(self.num_frames):
            clip_loss += self.clip_loss(results_final['img_frames'][idx], results_init['img_frames'][idx])
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
            points_scale = [self.config.deca_size, self.config.deca_size]  # Increases scale and shifts right + bottom for large values
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
        self.exp_codes = self.exp_codes.to(self.device)
        self.pose_codes = self.pose_codes.to(self.device)
        self.normalized_scores = self.normalized_scores.to(self.device)
        results_init, results_final = self.forward(batch)
        clip_loss = 0
        for idx in range(self.num_frames):
            clip_loss += self.clip_loss(results_final['img_frames'][idx], results_init['img_frames'][idx])
        self.log(f"val/clip_loss", clip_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        torch.cuda.empty_cache()

    @rank_zero_only
    def validation_epoch_end(self, _val_step_outputs):
        odir_texture, odir_samples, odir_ckpts = self.create_directories()
        self.export_images(odir_texture, odir_samples, odir_ckpts)

    def create_directories(self):
        output_dir_textures = Path(f'{self.config.base_dir}/runs/{self.config.experiment}/texture/')
        output_dir_images = Path(f'{self.config.base_dir}/runs/{self.config.experiment}/images/')
        output_dir_ckpts = Path(f'{self.config.base_dir}/runs/{self.config.experiment}/ckpts/{self.global_step:06d}')
        for odir in [output_dir_textures, output_dir_images, output_dir_ckpts]:
            odir.mkdir(exist_ok=True, parents=True)
        return output_dir_textures, output_dir_images, output_dir_ckpts

    def export_images(self, odir_texture, odir_samples, odir_ckpts):
        with torch.no_grad():
            for iter_idx, batch in enumerate(self.val_dataloader()):
                if iter_idx < self.config.num_vis_images:
                    batch['flame_shape'] = batch['flame_shape'].to(self.device)
                    batch['flame_exp'] = batch['flame_exp'].to(self.device)
                    batch['flame_pose'] = batch['flame_pose'].to(self.device)
                    batch['flame_cam'] = batch['flame_cam'].to(self.device)
                    self.exp_codes = self.exp_codes.to(self.device)
                    self.pose_codes = self.pose_codes.to(self.device)
                    self.normalized_scores = self.normalized_scores.to(self.device)

                    _, results_final = self.forward(batch)

                    odir_frame_images_pth = Path(f"{self.config.base_dir}/runs/{self.config.experiment}/frame_images/{self.global_step:06d}/{batch['name'][0]}")
                    odir_frame_texture_pth = Path(f"{self.config.base_dir}/runs/{self.config.experiment}/frame_texture/{self.global_step:06d}/{batch['name'][0]}")
                    odir_frame_images_pth.mkdir(exist_ok=True, parents=True)
                    odir_frame_texture_pth.mkdir(exist_ok=True, parents=True)
                    for frame_idx in range(self.num_frames):
                        save_image(results_final['img_frames'][frame_idx], odir_frame_images_pth / f"{frame_idx:02d}.png", value_range=(-1, 1), normalize=True)
                        save_image(results_final['tex_frames'][frame_idx], odir_frame_texture_pth / f"{frame_idx:02d}.png", value_range=(-1, 1), normalize=True)

                    vis_generated_images = torch.stack(results_final['img_frames'])[:, 0, :, :, :]
                    vis_generated_textures = torch.stack(results_final['tex_frames'])
                    torch.cuda.empty_cache()
                    save_image(vis_generated_images, odir_samples / f"{self.global_step:06d}_{iter_idx:02d}.png", nrow=int(math.sqrt(vis_generated_images.shape[0])), value_range=(-1, 1), normalize=True)
                    save_image(vis_generated_textures, odir_texture / f"{self.global_step:06d}_{iter_idx:02d}.png", nrow=int(math.sqrt(vis_generated_images.shape[0])), value_range=(-1, 1), normalize=True)

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
        self.G.to(self.device)
        self.flame.to(self.device)
        self.exp_codes.to(self.device)
        self.pose_codes.to(self.device)
        self.clip_loss.set_device(self.device)
        self.normalized_scores.to(self.device)
        for mapper in self.texture_mapper_list:
            mapper.to(self.device)
        self.train_set.set_device(self.device)
        self.val_set.set_device(self.device)


def step(opt, module_list):
    for module in module_list:
        for param in module.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        opt.step()


@hydra.main(config_path='../../config', config_name='clipface')
def main(config):
    trainer = create_trainer("StyleGANClipMLPVideo", config)
    model = StyleGANOptimizer(config)
    trainer.fit(model)


if __name__ == '__main__':
    main()
