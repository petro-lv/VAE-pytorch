import config
from model_parts import Encoder, Decoder, Discriminator

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, img_size, input_channels, latent_dim,
                 scaling_blocks, cnn_width):

        super(VAE, self).__init__()
        self.encoder = Encoder(img_size=img_size, in_channels=input_channels, latent_dim=latent_dim,
                               downscale_blocks=scaling_blocks, cnn_width=cnn_width)

        self.decoder = Decoder(img_size=img_size, out_channels=input_channels, latent_dim=latent_dim,
                               upscale_blocks=scaling_blocks, cnn_width=cnn_width)

    def forward(self, x):
        mu, log_sigma = self.encoder(x)

        z = self.reparameterize(mu, log_sigma)
        x_rec = self.decoder(z)
        return [mu, log_sigma], x_rec

    @staticmethod
    def reparameterize(mu, log_sigma):
        sigma = torch.exp(0.5 * log_sigma)
        eps = torch.randn(mu.shape, device=config.DEVICE)
        return mu + sigma * eps


vae = VAE(config.IMG_SIZE, 3, config.LATENT_DIM, config.SCALING_BLOCKS, config.MODEL_WIDTH).to(config.DEVICE)
discriminator = Discriminator(config.IMG_SIZE, 3).to(config.DEVICE)
