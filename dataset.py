import config

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose

import skimage.io as iio
from os import listdir, path


class Faces(Dataset):
    def __init__(self, root_dir):
        super(Faces, self).__init__()
        self.root_dir = root_dir
        self.images = listdir(root_dir)

        self.transform = Compose(
            [
                ToTensor(),
                Resize([config.IMG_SIZE, config.IMG_SIZE]),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im_path = path.join(self.root_dir, self.images[index])
        im = iio.imread(im_path)
        im = self.transform(im)
        return im
