import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(
        self, num_input_channels: int = 1, num_filters: int = 32, z_dim: int = 20
    ):
        """
        num_input_channels - Number of input channels of the image.
        num_filters - Number of channels we use in the first convolutional layers.
        z_dim - Dimensionality of latent representation z
        """
        super().__init__()

        c_hid = num_filters
        self.net = nn.Sequential(
            nn.Conv2d(
                num_input_channels, c_hid, kernel_size=3, padding=1, stride=2
            ),  # 32x32 => 16x16
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2
            ),  # 16x16 => 8x8
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2
            ),  # 8x8 => 4x4
            nn.GELU(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, z_dim),
        )
        self.proj_mean = nn.Linear(z_dim, z_dim)
        self.proj_log_std = nn.Linear(z_dim, z_dim)

    def forward(self, x):
        """
        x - Input batch with images of shape [B,C,H,W]
        """
        z = self.net(x)
        return z


class CNNDecoder(nn.Module):
    def __init__(
        self, num_input_channels: int = 16, num_filters: int = 32, z_dim: int = 20
    ):
        """
        num_input_channels - Number of channels of the image to reconstruct.
        num_filters - Number of filters we use in the last convolutional layers.
        z_dim - Dimensionality of latent representation z
        """
        super().__init__()

        self.num_input_channels = num_input_channels
        c_hid = num_filters
        self.linear = nn.Sequential(nn.Linear(z_dim, 2 * 16 * c_hid), nn.GELU())

        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid,
                2 * c_hid,
                kernel_size=3,
                output_padding=0,
                padding=1,
                stride=2,
            ),  # 4x4 => 7x7
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(
                2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 7x7 => 14x14
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(
                c_hid,
                num_input_channels,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),  # 14x14 => 28x28
        )

    def forward(self, z):
        """
        z - Latent vector of shape [B,z_dim]
        """
        x = self.linear(z)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


if __name__ == "__main__":
    batch_size = 16
    num_channels = 3
    w = h = 32

    num_filters = 32
    z_dim = 20
    enc = CNNEncoder(num_channels, num_filters, z_dim)
    dec = CNNDecoder(num_channels, num_filters, z_dim)

    x = torch.rand((batch_size, num_channels, h, w))
    z = enc(x)
    y = dec(z)

    print(y.shape)
    assert y.shape == (batch_size, num_channels, 28, 28)
