import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 transposed=False, apply_act=True):

        super(CNNBlock, self).__init__()

        if transposed:
            self.cnn = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding)

        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.relu = nn.LeakyReLU(0.1)
        self.apply_act = apply_act

    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)

        if self.apply_act:
            x = self.relu(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cnn1 = CNNBlock(in_channels, out_channels, 3, 1, 1)
        self.cnn2 = CNNBlock(out_channels, out_channels, 3, 1, 1)
        self.dim_equalizer = CNNBlock(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        initial = x

        x = self.cnn1(x)
        x = self.cnn2(x)

        if self.in_channels != self.out_channels:
            initial = self.dim_equalizer(initial)

        return x + initial


class Encoder(nn.Module):
    def __init__(self, img_size, in_channels,
                 latent_dim, downscale_blocks, cnn_width):

        super(Encoder, self).__init__()

        self.downscale_blocks = downscale_blocks
        self.latent_dim = latent_dim

        self.cnn, out_pixels = self._build_cnn_layers(
            img_size, cnn_width, in_channels
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(out_pixels, latent_dim)
        self.fc_sigma = nn.Linear(out_pixels, latent_dim)

    def _build_cnn_layers(self, img_size, cnn_width, in_channels):
        layers = []
        cnn_width = cnn_width // (2 ** self.downscale_blocks)
        for block in range(self.downscale_blocks):
            layers.append(ResBlock(in_channels, cnn_width))
            in_channels = cnn_width
            layers.append(nn.MaxPool2d(2))
            cnn_width *= 2

        layers.append(CNNBlock(in_channels, cnn_width, 1, 1, 0))

        res_size = img_size // (2 ** self.downscale_blocks)
        out_pixels_count = (res_size ** 2) * cnn_width

        return nn.Sequential(*layers), out_pixels_count

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        log_sigma = self.fc_sigma(x)
        return mu, log_sigma


class Decoder(nn.Module):
    def __init__(self, img_size, out_channels,
                 latent_dim, upscale_blocks, cnn_width):

        super(Decoder, self).__init__()
        self.initial_size = img_size // (2 ** upscale_blocks)
        self.out_channels = out_channels

        self.fc = nn.Linear(latent_dim, (self.initial_size ** 2) * 16)

        self.up_conv = self._build_upscaler(cnn_width, upscale_blocks)

    def _build_upscaler(self, cnn_width, upscale_blocks):
        layers = []
        in_channels = 16
        out_channels = cnn_width

        for _ in range(upscale_blocks):
            layers.append(
                CNNBlock(in_channels, out_channels, 2, 2, 0, transposed=True),
            )

            for _ in range(2):
                layers.append(
                    CNNBlock(out_channels, out_channels, 3, 1, 1)
                )
            in_channels = out_channels
            out_channels = out_channels // 2

        layers.append(
            CNNBlock(in_channels, self.out_channels, 1, 1, 0, apply_act=False)
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(-1, 16, self.initial_size, self.initial_size)
        x = self.up_conv(x)
        return torch.sigmoid(x)


class Discriminator(nn.Module):
    def __init__(self, img_size, in_channels):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels

        self.cnn, out_pixels = self._build_cnn_layers(img_size)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_pixels, 1)

    def _build_cnn_layers(self, img_size):
        architecture = [128, 64, 32, 16]
        layers = []

        cnn_in = self.in_channels
        for out_channels in architecture:
            layers.append(CNNBlock(cnn_in, out_channels, 3, 1, 1))
            cnn_in = out_channels

        layers.append(CNNBlock(architecture[-1], 1, 3, 1, 1))
        out_pixels = (img_size ** 2)

        return nn.Sequential(*layers), out_pixels

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return torch.sigmoid(x)
