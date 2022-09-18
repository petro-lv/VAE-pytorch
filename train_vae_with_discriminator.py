import config
from dataset import Faces
from models import vae, discriminator
from utils import vae_loss, save_random_result


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from tqdm import tqdm


ds = Faces(config.ROOT_DIR)

samples_count = len(ds)


loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True)

mse = nn.MSELoss()
ce_loss = nn.BCELoss()

decay = 0.03 * (config.BATCH_SIZE / (samples_count * config.NUM_EPOCHS))
optimizer = optim.Adam(vae.parameters(), lr=config.LEARNING_RATE, weight_decay=decay)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, weight_decay=decay)


if config.LOAD:
    state = torch.load(config.MODEL_SAVE_DIR)
    vae.encoder.load_state_dict(state['encoder'])
    vae.decoder.load_state_dict(state['decoder'])
    discriminator.load_state_dict(state['discriminator'])
    optimizer.load_state_dict(state['vae_optimizer'])
    discriminator_optimizer.load_state_dict(state['discriminator_optimizer'])


for epoch in range(config.NUM_EPOCHS):
    for im in tqdm(loader):
        im = im.to(config.DEVICE)
        encoded, reconstructed = vae(im)
        sample = torch.randn([config.BATCH_SIZE, config.LATENT_DIM], device=config.DEVICE)

        disc_real = discriminator(im)
        disc_fake = discriminator(reconstructed.detach())
        disc_sample = discriminator(vae.decoder(sample))

        disc_real_loss = ce_loss(disc_real, torch.ones_like(disc_real))
        disc_fake_loss = ce_loss(disc_fake, torch.zeros_like(disc_fake))
        disc_sample_loss = ce_loss(disc_sample, torch.zeros_like(disc_sample))

        disc_total_loss = disc_fake_loss + disc_real_loss + disc_sample_loss

        discriminator_optimizer.zero_grad()
        disc_total_loss.backward()
        discriminator_optimizer.step()

        mu, log_sigma = encoded

        disc_fake_rec = discriminator(reconstructed)
        adversarial_loss = ce_loss(disc_fake_rec, torch.ones_like(disc_fake_rec))

        loss = vae_loss(im, mu, log_sigma, reconstructed) + config.DISC_IMPORTANCE * adversarial_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save({
        'encoder': vae.encoder.state_dict(),
        'decoder': vae.decoder.state_dict(),
        'vae_optimizer': optimizer.state_dict(),
        'discriminator': discriminator.state_dict(),
        'discriminator_optimizer': discriminator_optimizer.state_dict()
    }, config.MODEL_SAVE_DIR)

    save_random_result(im.detach(), reconstructed.detach(), epoch)
