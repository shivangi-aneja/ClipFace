import torch
from model.stylegan_ada import FullyConnectedLayer, normalize_2nd_moment


class Mapper(torch.nn.Module):
    """ MLP-based mapper network. """

    def __init__(self, z_dim, w_dim, num_layers=8, activation='lrelu', lr_multiplier=0.01):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_layers = num_layers
        features_list = [z_dim] + [w_dim] * num_layers

        self.layers = torch.nn.ModuleList()
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            self.layers.append(FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier))

    def forward(self, z):
        # Embed, normalize, and concat inputs.
        x = None

        if self.z_dim > 0:
            x = normalize_2nd_moment(z)

        # Main layers.
        for idx in range(self.num_layers):
            x = self.layers[idx](x)

        return x
