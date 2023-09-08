import io
import random
from functools import partial
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torchvision
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, transforms=None):
        if isinstance(dir_path, str):
            self.dir_path = Path(dir_path)
        else:
            self.dir_path = dir_path

        self.raw_file_paths = [
            f for f in (dir_path / "raw").iterdir() if f.suffix in [".jpg"]
        ]
        self.jpeg_file_paths = [
            f for f in (dir_path / "jpeg").iterdir() if f.suffix in [".jpg"]
        ]
        assert len(self.raw_file_paths) == len(self.jpeg_file_paths)
        for i in tqdm(range(len(self.raw_file_paths))):
            assert self.raw_file_paths[i].stem == self.jpeg_file_paths[i].stem

        self.transforms = transforms


    def __getitem__(self, index):
        raw_file_path = self.raw_file_paths[index]
        jpeg_file_path = self.jpeg_file_paths[index]

        raw = Image.open(raw_file_path).convert('RGB')
        jpeg = Image.open(jpeg_file_path).convert('RGB')

        if self.transforms is not None:
            imgs = self.transforms([raw, jpeg])
            raw = imgs[0]
            jpeg = imgs[1]

        to_tensor = torchvision.transforms.ToTensor()
        raw = to_tensor(raw)
        jpeg = to_tensor(jpeg)

        return {"raw": raw, "jpeg": jpeg, "input_name": raw_file_path.stem}

    def __len__(self):
        return len(self.raw_file_paths)