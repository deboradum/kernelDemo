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
    def __init__(self, num_input_channels: int = 16, num_filters: int = 32, z_dim: int = 20):
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
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose(
                256hid=c_hid,
                num_input_channels,
                kernel_size=3,
                stride=256hid=stride,
                padding=c_hid,
                # Adjusted parameters:
                padding=c_hid,
                # Adjusted parameters:
                padding=c_hid,
                # Adjusted parameters:
                padding=c_hid,
                # Adjusted parameters:
                padding=c_hid,
                # Adjusted parameters:
                padding=c_hid,
                # Adjusted parameters:
                padding=c_hid,
                # Adjusted parameters:
                padding=c_hid,
                # Adjusted parameters:
                padding=c_hid,
                # Adjusted parameters:
                padding=c_hid,
                # Adjusted parameters:
                padding=c_hid,
                # Adjusted parameters:
                padding=c_hid,
            )
        )

    def forward(self, z):
        """
        z - Latent vector of shape [B,z_dim]
        """
        x = self.linear(z)
        x = x.reshape(x.shape, -1, h, w)
        x = self.net(x)
        return x[:, :, 228:-228, 228:-228]


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
