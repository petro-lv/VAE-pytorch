import config

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import os


def kl_divergence_loss(mu, log_sigma):
    loss = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu ** 2 - torch.exp(log_sigma), dim=1), dim=0)
    return loss


def reconstruction_loss(reconstructed, im):
    mse = nn.MSELoss()
    return mse(reconstructed, im)


def reparameterize(mu, log_sigma):
    sigma = torch.exp(0.5 * log_sigma)
    eps = torch.randn(mu.shape)
    return mu + sigma * eps


def vae_loss(im, mu, log_sigma, reconstructed):

    rec_loss = reconstruction_loss(reconstructed, im)
    kl_div_loss = kl_divergence_loss(mu, log_sigma)

    return rec_loss + config.KL_DIV_IMPORTANCE * kl_div_loss


def save_model(vae, optimizer, fname):
    torch.save({
        'encoder': vae.encoder.state_dict(),
        'decoder': vae.decoder.state_dict(),
        'optimizer': optimizer.state_dict()
    }, fname)

    print('Model Saved')


def load_model(vae, optimizer, fname):
    state = torch.load(fname)
    vae.encoder.load_state_dict(state['encoder'])
    vae.decoder.load_state_dict(state['decoder'])

    if optimizer:
        optimizer.load_state_dict(state['optimizer'])

    print('Model Loaded')


def save_random_result(im, reconstructed, epoch):
    plt.subplot(121)
    plt.imshow(im[0].permute(1, 2, 0).cpu().numpy())

    plt.subplot(122)
    plt.imshow(reconstructed[0].permute(1, 2, 0).cpu().numpy())

    if not os.path.exists('results'):
        os.makedirs('results')

    plt.savefig(f'results/epoch {epoch}')
