# -*- coding: utf-8 -*-
# ---------------------

import itertools
import random
import time
from typing import Union

import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from conf import Conf


def imread(img_path, colorspace='RGB', pil=False):
    # type: (str, str, bool) -> Union[np.ndarray, Image.Image]
    """
    Read image from path `img_path`
    :param img_path: path of the image to read
    :param colorspace: colorspace of the output image; must be one of {'RGB', 'BGR'}
    :param pil: if `True`, return a PIL.Image object, otherwise return a numpy array
    :return: PIL.Image object or numpy array in the specified colorspace
    """
    if pil:
        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as img:
                return img.convert('RGB')
    else:
        img = cv2.imread(img_path)
        if colorspace == 'RGB':
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class AutoencDS(Dataset):
    """
    Dataset composed of pairs (x, y) in which:
    * x: RGB image of size 128x128 representing a light blue circle (radius = 16 px)
         on a dark background (random circle position, randomly colored dark background)
    * y: copy of x, with the light blue circle surrounded with a red line (4 px internale stroke)
    """


    def __init__(self, cnf, mode, in_size=128, cut_size=2048):
        # type: (Conf, str, int, int) -> None
        """
        :param cnf: configuration object
        :param ds_len: dataset length
        """
        self.cnf = cnf
        self.mode = mode
        self.in_size = in_size

        self.counter = 0
        self.full_imgs = []
        self.random = np.random if mode == 'test' else np.random.RandomState(42)

        counter_mul = 8 if self.mode == 'train' else 2
        for img_path in itertools.chain(cnf.ds_path.files('*.png'), cnf.ds_path.files('*.jpg')):
            print(f'\r$> loading image files into memory, please wait... | \'{img_path.basename()}\'', end='')
            if self.mode == 'train':
                img = imread(img_path)[:, :-256, ...]
            else:
                img = imread(img_path)[:, -256:, ...]
            self.counter += counter_mul * (img.shape[0] // in_size) * (img.shape[1] // in_size)
            self.full_imgs.append(img)
        print('\r', end='')

        self.to_tensor = transforms.ToTensor()


    @property
    def n_surces(self):
        # type: () -> int
        return len(self.full_imgs)


    def __len__(self):
        # type: () -> int
        return self.counter


    def __getitem__(self, i):
        # type: (int) -> Tensor

        # select full image (random)
        idx = self.random.randint(0, len(self.full_imgs))
        full_img = self.full_imgs[idx]

        # random flip/rotate full image
        if self.random.random() < 0.5:
            full_img = np.fliplr(full_img)
        if self.cnf.flup_ud:
            if self.random.random() < 0.5:
                full_img = np.flipud(full_img)
            full_img = np.rot90(full_img, k=self.random.randint(0, 4))

        # select random cut
        r = self.random.randint(0, full_img.shape[0] - self.in_size)
        c = self.random.randint(0, full_img.shape[1] - self.in_size)
        cut = full_img[r:r + self.in_size, c:c + self.in_size].copy()

        # cut = self.augmenter.apply(img=cut, mode=self.mode)
        cut = transforms.ToTensor()(cut)
        return cut


    @staticmethod
    def wif(worker_id):
        # type: (int) -> None
        """
        Worker initialization function: set random seeds
        :param worker_id: worker int ID
        """
        seed = (int(round(time.time() * 1000)) + worker_id) % (2 ** 32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


    @staticmethod
    def wif_test(worker_id):
        # type: (int) -> None
        """
        Worker initialization function: set random seeds
        :param worker_id: worker int ID
        """
        seed = worker_id + 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def main():
    ds = AutoencDS(cnf=Conf(conf_file_path='/home/manicardi/Documenti/tesi/PixelCNN/conf/default.yaml', exp_name='toy'), mode='train')

    for i in range(len(ds)):
        x = ds[i]
        print(f'$> Example #{i}: x.shape {x.shape}')


if __name__ == '__main__':
    main()
