import torch
import torchvision
import torchvision.transforms.functional as F


class Resize(torchvision.transforms.Resize):
    def forward(self, img):
        if isinstance(img, list):
            return [self(one_img) for one_img in img]
        else:
            return self(img)


class RandomResizedCrop(torchvision.transforms.RandomResizedCrop):
    def forward(self, img):
        if isinstance(img, list):
            i, j, h, w = self.get_params(img[0], self.scale, self.ratio)

            return [
                F.resized_crop(one_img, i, j, h, w, self.size, self.interpolation)
                for one_img in img
            ]
        else:
            return self(img)

