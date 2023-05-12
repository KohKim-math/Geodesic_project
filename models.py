import torch
import torch.nn as nn
from torch_topological.nn import VietorisRipsComplex, SignatureLoss
from loss import DeviationFromIsometry
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, activation='ReLU'):
        super(Autoencoder, self).__init__()
        if activation == 'ReLU':
            nonlinear = nn.ReLU()
        if activation == 'ELU':
            nonlinear = nn.ELU()
        if activation == 'Tanh':
            nonlinear = nn.Tanh()

        self.rec_loss = nn.MSELoss()    
        self.encoder = nn.Sequential(
            nn.Linear(3, 10),
            nonlinear,
            nn.Linear(10, 5),
            nonlinear,
            nn.Linear(5, 2),
            )
        self.decoder = nn.Sequential(
            nn.Linear(2, 5),
            nonlinear,
            nn.Linear(5, 10),
            nonlinear,
            nn.Linear(10, 3),
            )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        rec_loss = self.rec_loss(x, x_hat)
        return rec_loss
    
class VAE(nn.Module):
    def __init__(self, activation='ReLU', latent_dim=2):
        super(VAE, self).__init__()
        if activation == 'ReLU':
            nonlinear = nn.ReLU()
        if activation == 'ELU':
            nonlinear = nn.ELU()
        if activation == 'Tanh':
            nonlinear = nn.Tanh()

        self.rec_loss = nn.MSELoss()    
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(3, 5),
            nonlinear,
            nn.Linear(5, 10),
            nonlinear,
            nn.Linear(10, 5),
            nonlinear,
            nn.Linear(5, self.latent_dim * 2) # output both mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 5),
            nonlinear,
            nn.Linear(5, 10),
            nonlinear,
            nn.Linear(10,5),
            nonlinear,
            nn.Linear(5, 3),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1) # split into mean and log-variance
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # compute the standard deviation
        eps = torch.randn_like(std) # generate a random noise tensor
        z = mu + eps * std # reparameterize the latent variable
        return z

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        rec_loss = self.rec_loss(x, x_hat)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) # compute KL divergence
        return rec_loss + 0.05*kl_loss
    

class GeoTopAutoencoder(torch.nn.Module):
    """Wrapper for a topologically-regularised autoencoder.

    This class uses another autoencoder model and imbues it with an
    additional topology-based loss term.
    """
    def __init__(self, model, lam_geo=1.0, lam_top=1.0, isVAE=False):
        super().__init__()

        self.lam_geo = lam_geo
        self.lam_top = lam_top
        self.model = model
        self.top_loss = SignatureLoss(p=2, normalise=False, dimensions=0)
        # self.geo_loss1 = DeviationFromIsometry(model.encoder)
        self.geo_loss2 = DeviationFromIsometry(model.decoder)
        self.isVAE = isVAE
        self.vr = VietorisRipsComplex(dim=0)

    def forward(self, x):
        z = self.model.encoder(x)
        if self.isVAE:
            mu, logvar = torch.chunk(z, 2, dim=-1) # split into mean and log-variance
            z = self.model.reparameterize(mu, logvar)
        recon_loss = self.model(x)

        if self.lam_top != 0:
            pi_x = self.vr(x)
            pi_z = self.vr(z)
            topo_loss = self.lam_top*self.top_loss([x, pi_x], [z, pi_z])/len(z) # divide by batch size
        else:
            topo_loss = torch.tensor([0.])


        if self.geo_loss2 != 0.:
            geoloss2 = self.geo_loss2(z)
            geo_loss = self.lam_geo*(geoloss2)
        else:
            geo_loss = torch.tensor([0.])
        
        loss = recon_loss + topo_loss + geo_loss
        loss_dict = {'loss' : [loss.item()], 'recon' : [recon_loss.item()], 'geo' : [geo_loss.item()], 'top' : [topo_loss.item()]}

        return loss, loss_dict