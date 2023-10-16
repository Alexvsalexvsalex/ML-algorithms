import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# Task 1

class Encoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, start_channels=16, downsamplings=5):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=start_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=(2 ** i) * start_channels,
                            out_channels=(2 ** (i + 1)) * start_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1
                        ),
                        nn.BatchNorm2d(num_features=(2 ** (i + 1)) * start_channels),
                        nn.ReLU()
                    )
                    for i in range(downsamplings)
                ]
            ),
            nn.Flatten(),
            nn.Linear(start_channels * (img_size ** 2) // (2 ** downsamplings), 128),
            nn.ReLU(),
            nn.Linear(128, 2 * latent_size)
        )

    def forward(self, x):
        res = self.net(x)
        mu, right = torch.split(res, self.latent_size, dim=1)
        sigma = torch.exp(right)
        r = torch.randn(self.latent_size, dtype=sigma.dtype).cuda()
        z = mu + sigma * r
        return z, (mu, sigma)


# Task 2

class Decoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, end_channels=16, upsamplings=5):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, end_channels * (img_size ** 2) // (2 ** upsamplings)),
            nn.Unflatten(
                dim=1,
                unflattened_size=(
                    2 ** upsamplings * end_channels,
                    img_size // (2 ** upsamplings),
                    img_size // (2 ** upsamplings)
                )
            ),
            nn.Sequential(
                *[
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels=end_channels * (2 ** i),
                            out_channels=end_channels * (2 ** (i - 1)),
                            kernel_size=4,
                            stride=2,
                            padding=1
                        ),
                        nn.BatchNorm2d(num_features=end_channels * (2 ** (i - 1))),
                        nn.ReLU()
                    )
                    for i in range(upsamplings, 0, -1)
                ]
            ),
            nn.Conv2d(
                in_channels=end_channels,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


# Task 3

class VAE(nn.Module):
    def __init__(self, img_size=128, downsamplings=5, latent_size=256, down_channels=8, up_channels=12):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_size, latent_size, down_channels, downsamplings)
        self.decoder = Decoder(img_size, latent_size, up_channels, downsamplings)

    def forward(self, x):
        z, (mu, sigma) = self.encoder(x)
        x_pred = self.decoder(z)
        kld = 0.5 * (torch.square(sigma) + torch.square(mu) - 2 * torch.log(sigma) - torch.ones(mu.shape).cuda())
        return x_pred, kld

    def encode(self, x):
        res, _ = self.encoder(x)
        return res

    def decode(self, z):
        return self.decoder(z)

    def save(self):
        torch.save(self.state_dict(), __file__[:-7] + 'vae.pth')

    def load(self):
        self.load_state_dict(torch.load(__file__[:-7] + 'vae.pth'))
        self.eval()
