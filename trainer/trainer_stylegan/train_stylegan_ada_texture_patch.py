import random
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')    # a fix for the "OSError: too many files" exception
import torch
import hydra
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torchvision.utils import save_image
from dataset.face_image import FaceImage
from model.stylegan_ada.augment import AugmentPipe
from model.stylegan_ada.generator import Generator
from model.stylegan_ada.discriminator import Discriminator
from model.renderer.differentiable_renderer import DifferentiableRenderer
from model.stylegan_ada.loss import PathLengthPenalty, compute_gradient_penalty, compute_gradient_penalty_patch
from trainer import create_trainer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
HYDRA_FULL_ERROR=1


class StyleGAN2Trainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.R = None
        self.G = Generator(z_dim=config.latent_dim, w_dim=config.latent_dim, w_num_layers=config.num_mapping_layers,
                           img_resolution=config.image_size, img_channels=3, synthesis_layer=config.generator)
        self.D = Discriminator(img_resolution=config.image_size, img_channels=3, w_num_layers=None)
        self.D_patch = Discriminator(img_resolution=config.patch_size, img_channels=3, w_num_layers=None)
        self.augment_pipe = AugmentPipe(config.ada_start_p, config.ada_target, config.ada_interval,
                                        config.ada_fixed, config.batch_size)
        self.grid_z = torch.randn(config.num_eval_images, self.config.latent_dim)
        self.train_set = FaceImage(config=self.config)
        self.val_set = FaceImage(config=self.config)
        self.automatic_optimization = False
        self.path_length_penalty = PathLengthPenalty(0.01, 2)
        self.ema = None
        if config.batch_gpu is None:
            config.batch_gpu = config.batch_size
        print(f"batch_size = {config.batch_size} / {config.batch_gpu}")
        assert config.batch_size >= config.batch_gpu and config.batch_size % config.batch_gpu == 0

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(list(self.G.parameters()), lr=self.config.lr_g, betas=(0.0, 0.99), eps=1e-8)
        d_opt = torch.optim.Adam(list(self.D.parameters()) + list(self.D_patch.parameters()), lr=self.config.lr_d,
                                 betas=(0.0, 0.99), eps=1e-8)
        return g_opt, d_opt

    def forward(self, batch, limit_batch_size=False):
        real = batch['image']
        z = self.latent(limit_batch_size)
        w = self.get_mapped_latent(z, 0.9)
        predicted_texture = self.G.synthesis(w)
        fake = self.render(predicted_texture, batch)
        return {"predicted_texture": predicted_texture, "real": real, "fake": fake, "w": w}

    def training_step(self, batch, batch_idx):
        total_acc_steps = self.config.batch_size // self.config.batch_gpu
        g_opt, d_opt = self.optimizers()
        b_size = batch['image'].shape[0]
        num_patches = self.config.num_patches
        # optimize generator
        g_opt.zero_grad(set_to_none=True)
        log_gen_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        log_gen_patch = torch.tensor(0, dtype=torch.float32, device=self.device)
        log_gen_total = torch.tensor(0, dtype=torch.float32, device=self.device)
        for acc_step in range(total_acc_steps):
            output = self.forward(batch=batch)
            fake = output["fake"]
            p_fake = self.D(self.augment_pipe(fake))
            fake_patches = fake.unfold(2, self.config.patch_size, self.config.patch_stride).unfold(3, self.config.patch_size, self.config.patch_stride).reshape(b_size, 3, -1, self.config.patch_size, self.config.patch_size)
            num_gen_patches = fake_patches.shape[2]
            patch_idxs = random.sample(range(num_gen_patches), num_patches)
            fake_patch_loss = 0
            for idx in patch_idxs:
                p_fake_patch = self.D_patch(self.augment_pipe(fake_patches[:, :, idx, :, :]))
                fake_patch_loss += torch.nn.functional.softplus(-p_fake_patch).mean()
            gen_loss = torch.nn.functional.softplus(-p_fake).mean()
            gen_loss_patch = fake_patch_loss / num_gen_patches
            gen_loss_total = gen_loss + gen_loss_patch

            # Backward pass on normal and patch loss
            self.manual_backward(gen_loss_total)

            log_gen_loss += gen_loss.detach()
            log_gen_patch += gen_loss_patch.detach()
            log_gen_total += gen_loss_total.detach()
        g_opt.step()
        log_gen_loss /= total_acc_steps
        log_gen_patch /= total_acc_steps
        log_gen_total /= total_acc_steps
        self.log("train/g_loss", log_gen_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/g_loss_patch", log_gen_patch, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/g_loss_total", log_gen_total, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        if self.global_step > self.config.lazy_path_penalty_after and (self.global_step + 1) % self.config.lazy_path_penalty_interval == 0:
            g_opt.zero_grad(set_to_none=True)
            log_plp_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            for acc_step in range(total_acc_steps):
                output = self.forward(batch=batch)
                fake = output["fake"]
                w = output["w"]
                plp = self.path_length_penalty(fake, w)
                if not torch.isnan(plp):
                    plp_loss = self.config.lambda_plp * plp * self.config.lazy_path_penalty_interval
                    self.manual_backward(plp_loss)
                    log_plp_loss += plp.detach()
            g_opt.step()
            log_plp_loss /= total_acc_steps
            self.log("train/rPLP", log_plp_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)

        # optimize discriminator
        d_opt.zero_grad(set_to_none=True)
        log_real_patch = torch.tensor(0, dtype=torch.float32, device=self.device)
        log_real_total = torch.tensor(0, dtype=torch.float32, device=self.device)
        log_real_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        log_fake_patch = torch.tensor(0, dtype=torch.float32, device=self.device)
        log_fake_total = torch.tensor(0, dtype=torch.float32, device=self.device)
        log_fake_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        masked_image = batch['image']
        batch_image = masked_image.split(self.config.batch_gpu)

        for acc_step in range(total_acc_steps):
            output = self.forward(batch=batch)
            fake = output["fake"]
            p_fake = self.D(self.augment_pipe(fake.detach()))

            fake_patches = fake.unfold(2, self.config.patch_size, self.config.patch_stride).unfold(3, self.config.patch_size, self.config.patch_stride).reshape(b_size, 3, -1, self.config.patch_size, self.config.patch_size)
            num_gen_patches = fake_patches.shape[2]
            patch_idxs = random.sample(range(num_gen_patches), num_patches)
            fake_patch_loss = 0
            for idx in patch_idxs:
                p_fake_patch = self.D_patch(self.augment_pipe(fake_patches[:, :, idx, :, :]))
                fake_patch_loss += torch.nn.functional.softplus(p_fake_patch).mean()

            fake_loss = torch.nn.functional.softplus(p_fake).mean()
            fake_loss_patch = fake_patch_loss / num_gen_patches
            fake_loss_total = fake_loss + fake_loss_patch

            self.manual_backward(fake_loss_total)

            log_fake_loss += fake_loss.detach()
            log_fake_patch += fake_loss_patch.detach()
            log_fake_total += fake_loss_total.detach()

            p_real = self.D(self.augment_pipe(batch_image[acc_step]))
            real_patches = batch_image[acc_step].unfold(2, self.config.patch_size, self.config.patch_stride).unfold(3, self.config.patch_size,self.config.patch_stride).reshape(b_size, 3, -1, self.config.patch_size, self.config.patch_size)
            num_gen_patches = real_patches.shape[2]
            real_patch_loss = 0
            for idx in patch_idxs:
                p_real_patch = self.D_patch(self.augment_pipe(real_patches[:, :, idx, :, :]))
                real_patch_loss += torch.nn.functional.softplus(-p_real_patch).mean()

            self.augment_pipe.accumulate_real_sign(p_real.sign().detach())

            real_loss = torch.nn.functional.softplus(-p_real).mean()
            real_loss_patch = real_patch_loss / num_gen_patches
            real_loss_total = real_loss + real_loss_patch

            self.manual_backward(real_loss_total)

            log_real_loss += real_loss.detach()
            log_real_patch += real_loss_patch.detach()
            log_real_total += real_loss_total.detach()

        d_opt.step()
        log_real_loss /= total_acc_steps
        log_real_patch /= total_acc_steps
        log_real_total /= total_acc_steps
        log_fake_loss /= total_acc_steps
        log_fake_patch /= total_acc_steps
        log_fake_total /= total_acc_steps

        self.log("train/D_real", log_real_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/D_real_patch", log_real_patch, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/D_fake", log_fake_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/D_fake_patch", log_fake_patch, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        disc_loss = log_real_loss + log_fake_loss
        disc_loss_patch = log_real_patch + log_fake_patch
        disc_loss_total = log_real_total + log_fake_total
        self.log("train/d_loss", disc_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/d_loss_patch", disc_loss_patch, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/disc_loss_total", disc_loss_total, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        if (self.global_step + 1) % self.config.lazy_gradient_penalty_interval == 0:
            d_opt.zero_grad(set_to_none=True)
            log_gp_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            log_gp_loss_patch = torch.tensor(0, dtype=torch.float32, device=self.device)
            masked_image = batch['image']
            # masked_image = F.interpolate(batch['image'], self.config.image_size)
            batch_image = masked_image.split(self.config.batch_gpu)
            for acc_step in range(total_acc_steps):
                # For Full Image
                batch_image[acc_step].requires_grad_(True)
                p_real = self.D(self.augment_pipe(batch_image[acc_step], disable_grid_sampling=True))
                gp = compute_gradient_penalty(batch_image[acc_step], p_real)
                gp_loss = self.config.lambda_gp * gp * self.config.lazy_gradient_penalty_interval
                self.manual_backward(gp_loss)
                log_gp_loss += gp.detach()

                # For atches
                real_patches = batch_image[acc_step].unfold(2, self.config.patch_size, self.config.patch_stride).unfold(3, self.config.patch_size, self.config.patch_stride).reshape(b_size, 3, -1, self.config.patch_size, self.config.patch_size)
                num_gen_patches = real_patches.shape[2]
                patch_idxs = random.sample(range(num_gen_patches), num_patches)

                for idx in patch_idxs:
                    p_real_patch = self.D_patch(self.augment_pipe(real_patches[:, :, idx, :, :], disable_grid_sampling=True))
                    gp_patch = compute_gradient_penalty_patch(real_patches, p_real_patch, idx)  # Get the gradient at the given index
                    gp_patch_loss = self.config.lambda_gp * gp_patch * self.config.lazy_gradient_penalty_interval
                    self.manual_backward(gp_patch_loss)
                    log_gp_loss_patch += gp_patch.detach()

            d_opt.step()
            log_gp_loss /= total_acc_steps
            self.log("train/rGP", log_gp_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log("train/rGP_patch", log_gp_loss_patch, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        self.execute_ada_heuristics()
        self.ema.update(self.G.parameters())

    def execute_ada_heuristics(self):
        if (self.global_step + 1) % self.config.ada_interval == 0:
            self.augment_pipe.heuristic_update()
        self.log("aug_p", self.augment_pipe.p.item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    def render(self, pred_texture, batch):
        vertices, uvs = [], []
        faces, uv_indices = [], []
        num_vertex, num_uvs = 0, 0
        start_index = 0
        ranges = []
        batch_size = batch['image'].shape[0]
        for i in range(batch_size):
            pred_verts = batch['vertices'][i]
            vertices.append(pred_verts)
            uvs.append(self.train_set.uvs)
            faces.append(self.train_set.faces + num_vertex)
            uv_indices.append(self.train_set.uv_indices + num_uvs)
            num_vertex += pred_verts.shape[0]
            num_uvs += self.train_set.uvs.shape[0]
            ranges.append(torch.tensor([start_index, self.train_set.faces.shape[0]]).int())
            start_index += self.train_set.faces.shape[0]
        vertices, uvs, faces, uv_indices, ranges = torch.cat(vertices, 0), torch.cat(uvs, 0), torch.cat(faces,0), torch.cat(uv_indices, 0), torch.stack(ranges, dim=0)
        rendered_img = self.R.render_with_texture_map(vertices, faces, uvs, uv_indices, pred_texture, ranges=ranges).permute((0, 3, 1, 2)).contiguous()
        return rendered_img

    def validation_step(self, batch, batch_idx):
        pass

    @rank_zero_only
    def validation_epoch_end(self, _val_step_outputs):
        odir_samples, odir_grid, odir_texture, odir_real, odir_fake, odir_ckpts = self.create_directories()
        self.export_images("", odir_grid, odir_samples, odir_texture, odir_fake)
        # Save the original G & D params
        torch.save(self.G.state_dict(), odir_ckpts / f"G_{self.global_step:06d}.pth")
        torch.save(self.D.state_dict(), odir_ckpts / f"D_{self.global_step:06d}.pth")
        torch.save(self.D_patch.state_dict(), odir_ckpts / f"D_patch_{self.global_step:06d}.pth")
        self.ema.store(self.G.parameters())
        self.ema.copy_to(self.G.parameters())
        self.export_images("ema_", odir_grid, None, None, None)
        # Save the exponentially averaged results
        torch.save(self.G.state_dict(), odir_ckpts / f"{self.global_step:06d}.pth")
        self.ema.restore(self.G.parameters())
        for iter_idx, batch in enumerate(self.val_dataloader()):
            if iter_idx < self.config.num_vis_images // self.config.batch_size:
                for batch_idx in range(batch['image'].shape[0]):
                    save_image(batch['image'][batch_idx], odir_real / f"{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
            else:
                break

    def get_mapped_latent(self, z, style_mixing_prob):
        if torch.rand(()).item() < style_mixing_prob:
            cross_over_point = int(torch.rand(()).item() * self.G.mapping.num_ws)
            w1 = self.G.mapping(z[0])[:, :cross_over_point, :]
            w2 = self.G.mapping(z[1], skip_w_avg_update=True)[:, cross_over_point:, :]
            return torch.cat((w1, w2), dim=1)
        else:
            w = self.G.mapping(z[0])
            return w

    def latent(self, limit_batch_size=False):
        batch_size = self.config.batch_gpu if not limit_batch_size else self.config.batch_gpu // self.path_length_penalty.pl_batch_shrink
        z1 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        z2 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        return z1, z2

    def train_dataloader(self):
        return DataLoader(self.train_set, self.config.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.config.batch_gpu, shuffle=False, drop_last=True, num_workers=self.config.num_workers)

    def export_images(self, prefix, output_dir_grid, output_dir_samples, output_dir_texture, output_dir_fid):
        vis_generated_images = []
        for iter_idx, batch in enumerate(self.val_dataloader()):
            if iter_idx < self.config.num_vis_images // self.config.batch_size:
                real = batch['image']
                batch['vertices'] = batch['vertices'].to(self.device)
                batch['image'] = batch['image'].to(self.device)
                z = self.grid_z.split(self.config.batch_gpu)[iter_idx].to(self.device)
                predicted_texture = self.G(z, noise_mode='const')
                fake = self.render(predicted_texture, batch)
                for batch_idx in range(fake.shape[0]):
                    if output_dir_samples is not None and output_dir_texture is not None and output_dir_fid is not None:
                        save_image(predicted_texture[batch_idx], output_dir_texture / f"tex_{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
                        save_image(real[batch_idx], output_dir_samples / f"real_{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
                        save_image(fake[batch_idx], output_dir_samples / f"pred_{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
                        save_image(fake[batch_idx], output_dir_fid / f"{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
                    vis_generated_images.append(fake[batch_idx].unsqueeze(0))
            else:
                break
        torch.cuda.empty_cache()
        vis_generated_images = torch.cat(vis_generated_images, dim=0)
        save_image(vis_generated_images, output_dir_grid / f"{prefix}{self.global_step:06d}.png", nrow=int(self.config.batch_size), value_range=(-1, 1), normalize=True)

    def create_directories(self):
        output_dir_grid = Path(f'{self.config.base_dir}/runs/{self.config.experiment}/grid/')
        output_dir_fid_real = Path(f'{self.config.base_dir}/runs/{self.config.experiment}/fid/real')
        output_dir_fid_fake = Path(f'{self.config.base_dir}/runs/{self.config.experiment}/fid/fake')
        output_dir_texture = Path(f'{self.config.base_dir}/runs/{self.config.experiment}/texture/{self.global_step:06d}')
        output_dir_samples = Path(f'{self.config.base_dir}/runs/{self.config.experiment}/images/{self.global_step:06d}')
        output_dir_ckpts = Path(f'{self.config.base_dir}/runs/{self.config.experiment}/ckpts/')
        for odir in [output_dir_grid, output_dir_samples, output_dir_texture, output_dir_fid_real, output_dir_fid_fake, output_dir_ckpts]:
            odir.mkdir(exist_ok=True, parents=True)
        return output_dir_samples, output_dir_grid, output_dir_texture, output_dir_fid_real, output_dir_fid_fake, output_dir_ckpts

    def on_train_start(self):
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.G.parameters(), 0.995)
        self.run_post_device_setup()

    def on_validation_start(self):
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.G.parameters(), 0.995)
        self.run_post_device_setup()

    def run_post_device_setup(self):
        if self.R is None:
            self.R = DifferentiableRenderer(self.config.image_size, "standard")
        self.R.to(self.device)
        self.G.to(self.device)
        self.D.to(self.device)
        self.D_patch.to(self.device)
        self.ema.to(self.device)
        self.train_set.set_device(self.device)
        self.val_set.set_device(self.device)


@hydra.main(config_path='../../config', config_name='stylegan_ada')
def main(config):
    trainer = create_trainer("StyleGAN-Ada-Tex", config)
    model = StyleGAN2Trainer(config)
    trainer.fit(model)


if __name__ == '__main__':
    main()
