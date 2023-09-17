import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from einops import repeat
from einops.layers.torch import Rearrange

from vector_quantize_pytorch import ResidualVQ
import modules

class RQVAE(nn.Module):
    def __init__(self,
        in_channels,
        embedding_dim,
        num_embeddings,
        hidden_dims=None,
        num_quantizers=8,
        img_size=64,
        **kwargs
    ):
        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
        )

        for _ in range(6):
            modules.append(modules.ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU()
            )
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = ResidualVQ(
                            dim=embedding_dim,
                            num_quantizers=num_quantizers,   # specify number of quantizers
                            codebook_size=num_embeddings,    # codebook size
                        )

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], out_channels=1, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
                nn.Upsample((28,28))
            )
        )

        self.decoder = nn.Sequential(*modules)
        
    def forward(self, x):
        encoding = self.encoder(x)
        img_h, img_w  = encoding.shape[2:]
        encoding_flat = rearrange(encoding, 'b c h w -> b (h w) c')
        quantized_inputs, indices, commit_loss = self.vq_layer(encoding_flat)
        quantized_inputs = rearrange(quantized_inputs, 'b (h w) c -> b c h w', h=img_h)
        return [self.decoder(quantized_inputs), x, commit_loss]
    
    def loss_function(self, recons_x, x, commit_loss):
        recons_loss = F.mse_loss(recons, input)
        loss = recons_loss + vq_loss.sum()
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}
    
    @torch.no_grad()
    def reconstruct(self, x):
        return self.forward(x)[0]

