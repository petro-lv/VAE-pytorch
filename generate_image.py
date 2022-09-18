import argparse
import torch
import numpy as np
import os
from PIL import Image

import config
from models import vae


parser = argparse.ArgumentParser()

parser.add_argument('--face', type=str, choices=['anime', 'real'], default='anime',
                    help='Face to generate')
parser.add_argument('--save-dir', type=str, default='generated_images',
                    help='Directory to save an image, will be created if doesn\'t exist')
parser.add_argument('--image-name', type=str, default='face',
                    help='Name of saved image in the directory')
parser.add_argument('-d', '--discriminator', action='store_true',
                    help='If specified, model trained with discriminator will be used')

args = parser.parse_args()

if args.face == 'anime':
    if args.discriminator:
        model_dir = 'trained_models/vae_anime_faces_with_disc.pth'
    else:
        model_dir = 'trained_models/vae_anime_faces.pth'
else:
    if args.discriminator:
        model_dir = 'trained_models/vae_real_faces_with_disc.pth'
    else:
        model_dir = 'trained_models/vae_real_faces.pth'

model_state = torch.load(model_dir)

vae.decoder.load_state_dict(model_state['decoder'])
vae.decoder.to('cpu')
vae.decoder.eval()

sample = torch.randn(config.LATENT_DIM)
image = vae.decoder(sample)

# transforming shape from (channels, height, width) to standard (height, width, channels)
image = image.squeeze(0).detach().permute(1, 2, 0).numpy()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

photo_save_dir = args.save_dir + '/' + args.image_name + '.png'

image = image * 255
image = image.astype(np.uint8)
image = Image.fromarray(image)
image.save(photo_save_dir)
