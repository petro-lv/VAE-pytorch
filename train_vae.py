import config
from dataset import Faces
from utils import vae_loss, save_model, load_model, save_random_result
from models import vae

import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm


ds = Faces(config.ROOT_DIR)

samples_count = len(ds)

loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True)


decay = 0.03 * (config.BATCH_SIZE / (samples_count * config.NUM_EPOCHS))
optimizer = optim.Adam(vae.parameters(), lr=config.LEARNING_RATE, weight_decay=decay)

if config.LOAD:
    load_model(vae, optimizer, config.MODEL_SAVE_DIR)

for epoch in range(config.NUM_EPOCHS):

    for im in tqdm(loader):
        im = im.to(config.DEVICE)
        encoded, reconstructed = vae(im)

        mu, log_sigma = encoded

        loss = vae_loss(im, mu, log_sigma, reconstructed)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    save_model(vae, optimizer, config.MODEL_SAVE_DIR)

    save_random_result(im.detach(), reconstructed.detach(), epoch)
