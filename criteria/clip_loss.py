import torch
import clip
from torchvision.transforms import transforms


class ClipLoss(torch.nn.Module):

    def __init__(self, config):
        super(ClipLoss, self).__init__()
        self.base_prompt = config.base_prompt
        self.altered_prompt = config.altered_prompt
        self.loss_type = config.clip_loss_type
        self.clip_model, _ = clip.load('ViT-B/32', jit=False)
        self.directional = config.clip_directional
        self.loss_func = {
            'mse':    torch.nn.MSELoss(),
            'cosine': torch.nn.CosineSimilarity(dim=1, eps=1e-6),
            'mae':    torch.nn.L1Loss()
        }[self.loss_type]
        clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                               (0.26862954, 0.26130258, 0.27577711))
        self.clip_transform = transforms.Compose([transforms.Resize((224, 224)), clip_normalizer])

    def set_device(self, device):
        self.device = device

    def get_text_features(self, text_prompt: str, norm: bool = True) -> torch.Tensor:
        tokens = clip.tokenize([text_prompt]).to(self.device)
        text_features = self.clip_model.encode_text(tokens).detach()
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        clip_image = self.clip_transform(torch.clamp(img * 0.5 + 0.5, 0, 1))  # Clip model expects image to be normalized to (0, 1)
        image_features = self.clip_model.encode_image(clip_image)
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features

    def compute_text_direction(self, source_text: str, target_text: str) -> torch.Tensor:
        source_features = self.get_text_features(source_text)
        target_features = self.get_text_features(target_text)
        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        return text_direction

    def compute_img2img_direction(self, source_img: torch.Tensor, target_imgs: torch.Tensor) -> torch.Tensor:
        src_encoding = self.get_image_features(source_img)
        target_encoding = self.get_image_features(target_imgs)
        img_direction = (target_encoding - src_encoding)
        img_direction /= (img_direction.clone().norm(dim=-1, keepdim=True) + 1e-6)
        return img_direction

    def forward(self, altered_images, init_images=None, weight=1):
        if self.directional:
            # Use directional clip loss
            img_direction = self.compute_img2img_direction(init_images, altered_images)
            text_direction = weight*self.compute_text_direction(self.base_prompt, self.altered_prompt)
            # Check for loss type
            if self.loss_type == "cosine":
                loss = 1. - self.loss_func(img_direction.double(), text_direction)
            else:
                loss = self.loss_func(img_direction, text_direction)
            return loss.mean()
        else:
            # Undirectional Clip loss
            prompt_token = clip.tokenize([self.altered_prompt]).to(self.device)
            encoded_text = self.clip_model.encode_text(prompt_token)
            clip_images = self.clip_transform(torch.clamp(altered_images * 0.5 + 0.5, 0, 1))
            encoded_renders = self.clip_model.encode_image(clip_images)

            # Check for loss type
            if self.loss_type == "cosine":
                return - torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))